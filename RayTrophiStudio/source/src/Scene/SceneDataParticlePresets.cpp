#include "scene_data.h"

// Embedded production presets stay out of SceneData's foundational header.
// This translation unit owns authored particle, gas, and fluid configurations;
// SceneData only exposes the small preset enum and dispatch method.

SceneData::ParticleSystemObject& SceneData::addParticleSystemPreset(
    SceneData::ParticleSystemPreset preset) {
        const std::size_t systems_before = particle_systems.size();
        const char* preset_name = "Particle System";
        switch (preset) {
            case ParticleSystemPreset::Campfire:    preset_name = "Campfire";     break;
            case ParticleSystemPreset::Explosion:   preset_name = "Explosion";    break;
            case ParticleSystemPreset::Smoke:       preset_name = "Smoke";        break;
            case ParticleSystemPreset::GroundBurst: preset_name = "Ground Burst"; break;
            case ParticleSystemPreset::Fireball:    preset_name = "Fireball";     break;
            case ParticleSystemPreset::Flamethrower:preset_name = "Flamethrower"; break;
            case ParticleSystemPreset::BurningFuelSpill:preset_name = "Burning Fuel Spill"; break;
            case ParticleSystemPreset::IgnitedFuelJet:preset_name = "Ignited Fuel Jet"; break;
            case ParticleSystemPreset::NuclearCinematic:preset_name = "Nuclear Detonation (Cinematic)"; break;
            case ParticleSystemPreset::NuclearPhysical:preset_name = "Nuclear Detonation (Physical)"; break;
        }
        // This replaces the old policy that avoided
        // spawning a brand-new system on every click — consecutive preset presses
        // Each click now creates a fresh runtime; existing systems are untouched.
        ParticleSystemObject& sys = addParticleSystemObject(preset_name);
        auto rt = sys.runtime;
        if (!rt) return sys;

        // The new runtime starts empty. Scene-wide rigid-body proxy colliders
        // installed by addParticleSystemObject are intentionally retained.
        sys.render = ParticleRenderSettings{};
        sys.name = std::string(preset_name) + " #" + std::to_string(sys.id);

        switch (preset) {
            case ParticleSystemPreset::Campfire: {
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Spark);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Realtime);
                rt->setGravity(Vec3(0.0f, -1.6f, 0.0f));   // gentle: sparks rise then drift down
                rt->setLinearDrag(0.5f);
                RayTrophiSim::ParticleEmitterDesc e;
                e.name = "Campfire Emitter";
                e.point = Vec3(0.0f, 0.1f, 0.0f);
                e.direction = Vec3(0.0f, 1.0f, 0.0f);
                e.rate_per_second = 70.0f;
                e.speed = 1.6f;
                e.spread = 0.35f;
                e.lifetime_seconds = 1.4f;
                e.start_size = 0.08f;  e.end_size = 0.01f;  e.size_jitter = 0.5f;
                e.start_opacity = 1.0f; e.end_opacity = 0.0f;
                e.start_color = Vec3(1.0f, 0.8f, 0.35f); e.end_color = Vec3(0.9f, 0.15f, 0.03f);
                e.angular_velocity = 2.0f; e.angular_jitter = 3.0f;
                rt->addEmitter(e);

                // Rising embers feed the plume they came from: a little fuel so
                // they keep the flame alive as they drift, plus heat so the gas
                // lifts around each one instead of ignoring them.
                rt->physicsSettings().grid_density_deposit = 0.8f;
                rt->physicsSettings().grid_temperature_deposit = 1.6f;
                rt->physicsSettings().grid_fuel_deposit = 0.45f;

                // Hybrid effect: sparks remain discrete RT particles while a
                // co-located Vulkan gas domain supplies flame and smoke.
                RayTrophiSim::SimulationGridDomainDesc dom;
                dom.name = "Campfire Gas";
                dom.backend = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                dom.boundary_mode = RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                dom.gas_maccormack_advection = true;
                dom.bounds_min = Vec3(-1.5f, 0.0f, -1.5f);
                dom.bounds_max = Vec3(1.5f, 4.0f, 1.5f);
                // voxel_size is the resolution authority: the domain sync
                // recomputes resolution_* from extent/voxel_size whenever
                // preserve_voxel_size_on_resize is set (the default), so writing
                // resolution_* here would simply be overwritten on frame 1.
                // 0.055 over 3x4x3 m -> ~55x73x55.
                dom.voxel_size = 0.055f;
                dom.channels |= static_cast<uint32_t>(
                    RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                dom.fire_enabled = true;
                dom.ignition_temperature = 0.2f;
                dom.burn_rate = 2.4f;
                dom.heat_release = 2.8f;
                dom.smoke_generation = 0.7f;
                dom.flame_dissipation = 2.4f;
                dom.gas_buoyancy_heat = 1.4f;
                dom.gas_buoyancy_density = 0.04f;
                dom.gas_vorticity = 0.55f;
                dom.fire_expansion = 0.02f;
                dom.turbulence_strength = 0.52f;
                dom.turbulence_scale = 2.0f;
                dom.turbulence_octaves = 4;
                dom.turbulence_persistence = 0.52f;
                dom.shader = VolumeShader::createFirePreset();
                rt->addGridDomain(dom);

                RayTrophiSim::SimulationFlowSourceDesc fire;
                fire.name = "Campfire Flame Source";
                fire.domain_index = 0;
                fire.position = e.point;
                fire.radius = 0.32f;
                fire.velocity = Vec3(0.0f, 1.1f, 0.0f);
                fire.density = 0.35f;
                fire.temperature = 1.2f;
                fire.fuel = 1.5f;
                fire.falloff = 1.5f;
                rt->addFlowSource(fire);

                sys.blend_mode = ParticleBlendMode::Additive;
                sys.render.render_in_raytrace = true;
                sys.render.shape = ParticleRenderShape::Sphere;
                sys.render.emissive = true;
                sys.render.base_color = Vec3(1.0f, 0.75f, 0.3f);
                sys.render.emission_strength = 8.0f;
                break;
            }
            case ParticleSystemPreset::Explosion: {
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Spark);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Realtime);
                rt->setGravity(Vec3(0.0f, -9.81f, 0.0f));
                rt->setLinearDrag(0.12f);
                rt->setCollisionPlane(0.0f, true, 0.3f);   // debris bounces on the ground
                RayTrophiSim::ParticleEmitterDesc e;
                e.name = "Explosion Burst";
                e.point = Vec3(0.0f, 1.0f, 0.0f);
                e.direction = Vec3(0.0f, 1.0f, 0.0f);
                e.rate_per_second = 0.0f;
                e.burst_count = 400;
                e.speed = 6.0f;
                e.spread = 3.0f;          // near-omnidirectional
                e.lifetime_seconds = 2.0f;
                e.start_size = 0.1f;  e.end_size = 0.04f;  e.size_jitter = 0.6f;
                e.start_opacity = 1.0f; e.end_opacity = 0.0f;
                e.start_color = Vec3(1.0f, 0.9f, 0.5f); e.end_color = Vec3(0.3f, 0.08f, 0.02f);
                e.angular_velocity = 4.0f; e.angular_jitter = 8.0f;
                rt->addEmitter(e);

                RayTrophiSim::ParticleEmitterDesc core = e;
                core.name = "Explosion Fireball Core";
                core.burst_count = 220;
                core.speed = 2.2f;
                core.spread = 2.8f;
                core.lifetime_seconds = 0.7f;
                core.start_size = 0.32f;
                core.end_size = 0.05f;
                core.size_jitter = 0.35f;
                core.start_color = Vec3(1.0f, 0.95f, 0.65f);
                core.end_color = Vec3(1.0f, 0.16f, 0.015f);
                core.angular_velocity = 1.5f;
                core.angular_jitter = 3.0f;
                core.seed = 0x51f15e5du;
                rt->addEmitter(core);

                // THE point of the preset: the shrapnel is burning. Each piece
                // drops fuel and heat into the gas along its arc, so the domain
                // ignites where the debris flies and the fireball spreads WITH
                // the scatter instead of being a static ball the debris exits.
                rt->physicsSettings().grid_density_deposit = 2.2f;
                rt->physicsSettings().grid_temperature_deposit = 6.0f;
                rt->physicsSettings().grid_fuel_deposit = 2.8f;

                // Short fuel/heat pulse drives a real volumetric blast; the
                // discrete burst remains the hot debris/spark layer.
                RayTrophiSim::SimulationGridDomainDesc dom;
                dom.name = "Explosion Gas";
                dom.backend = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                dom.boundary_mode = RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                dom.gas_maccormack_advection = true;
                dom.bounds_min = Vec3(-3.0f, 0.0f, -3.0f);
                dom.bounds_max = Vec3(3.0f, 6.0f, 3.0f);
                dom.voxel_size = 0.075f;   // 6 m box -> 80^3
                dom.channels |= static_cast<uint32_t>(
                    RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                dom.fire_enabled = true;
                dom.ignition_temperature = 0.1f;
                dom.burn_rate = 7.0f;
                dom.heat_release = 5.0f;
                dom.smoke_generation = 1.1f;
                dom.flame_dissipation = 1.8f;
                dom.fire_expansion = 0.85f;
                dom.gas_buoyancy_heat = 0.35f;
                dom.gas_buoyancy_density = 0.02f;
                dom.gas_vorticity = 0.8f;
                dom.turbulence_strength = 0.90f;
                dom.turbulence_scale = 2.6f;
                dom.turbulence_octaves = 5;
                dom.turbulence_lacunarity = 2.1f;
                dom.turbulence_persistence = 0.56f;
                dom.turbulence_speed = 1.35f;
                dom.shader = VolumeShader::createExplosionPreset();
                rt->addGridDomain(dom);

                RayTrophiSim::SimulationFlowSourceDesc blast;
                blast.name = "Explosion Fuel Pulse";
                blast.domain_index = 0;
                blast.position = e.point;
                blast.radius = 0.75f;
                blast.velocity = Vec3(0.0f, 0.5f, 0.0f);
                blast.density = 2.0f;
                blast.temperature = 5.0f;
                blast.fuel = 8.0f;
                blast.falloff = 0.6f;
                blast.use_time_limit = true;
                blast.start_time = 0.0f;
                blast.end_time = 0.12f;
                rt->addFlowSource(blast);

                sys.blend_mode = ParticleBlendMode::Additive;
                sys.render.render_in_raytrace = true;
                sys.render.shape = ParticleRenderShape::Tetra;  // chunky debris (or set SceneMeshes)
                sys.render.emissive = true;
                sys.render.base_color = Vec3(1.0f, 0.8f, 0.4f);
                sys.render.emission_strength = 6.0f;
                break;
            }
            case ParticleSystemPreset::Smoke: {
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Gas);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Preview);
                RayTrophiSim::SimulationGridDomainDesc dom;
                dom.name = "Smoke Domain";
                dom.backend = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                dom.boundary_mode = RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                dom.gas_maccormack_advection = true;
                dom.bounds_min = Vec3(-2.0f, 0.0f, -2.0f);
                dom.bounds_max = Vec3(2.0f, 5.0f, 2.0f);
                dom.voxel_size = 0.07f;    // 4x5x4 m -> ~57x71x57
                dom.fire_enabled = false;                       // smoke only, no combustion
                dom.gas_buoyancy_heat = 0.75f;
                dom.gas_buoyancy_density = 0.035f;
                dom.gas_vorticity = 0.48f;
                dom.turbulence_strength = 0.42f;
                dom.turbulence_scale = 1.45f;
                dom.turbulence_octaves = 4;
                dom.turbulence_persistence = 0.52f;
                dom.shader = VolumeShader::createSmokePreset();
                rt->addGridDomain(dom);
                RayTrophiSim::SimulationFlowSourceDesc fs;
                fs.name = "Smoke Source";
                fs.position = Vec3(0.0f, 0.3f, 0.0f);
                fs.velocity = Vec3(0.0f, 1.5f, 0.0f);
                fs.radius = 0.35f;
                fs.density = 1.0f;
                fs.temperature = 0.4f;
                rt->addFlowSource(fs);
                sys.render.render_in_raytrace = false;          // volumetric, drawn by the VDB bridge
                break;
            }
            case ParticleSystemPreset::GroundBurst: {
                // Ground detonation: the floor clips the blast, so energy that
                // would have gone downward is redirected outward and then up.
                // Debris is thrown low and wide, drags burning fuel through the
                // domain, and the column climbs behind it.
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Spark);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Realtime);
                rt->setGravity(Vec3(0.0f, -9.81f, 0.0f));
                rt->setLinearDrag(0.2f);
                rt->setCollisionPlane(0.0f, true, 0.25f);   // dirt skips along the ground

                // Low, wide shrapnel fan: spread stays under a hemisphere so the
                // cone hugs the ground instead of firing straight up.
                RayTrophiSim::ParticleEmitterDesc debris;
                debris.name = "Ground Debris";
                debris.point = Vec3(0.0f, 0.15f, 0.0f);
                debris.direction = Vec3(0.0f, 1.0f, 0.0f);
                debris.rate_per_second = 0.0f;
                debris.burst_count = 380;
                debris.speed = 7.5f;
                debris.spread = 1.45f;
                debris.lifetime_seconds = 2.6f;
                debris.start_size = 0.09f; debris.end_size = 0.03f; debris.size_jitter = 0.7f;
                debris.start_opacity = 1.0f; debris.end_opacity = 0.0f;
                debris.start_color = Vec3(1.0f, 0.72f, 0.28f);
                debris.end_color = Vec3(0.22f, 0.09f, 0.05f);
                debris.angular_velocity = 5.0f; debris.angular_jitter = 9.0f;
                rt->addEmitter(debris);

                // Slow, heavy dirt that arcs and falls back: mass, not fire.
                RayTrophiSim::ParticleEmitterDesc dirt = debris;
                dirt.name = "Thrown Dirt";
                dirt.burst_count = 260;
                dirt.speed = 4.0f;
                dirt.spread = 1.1f;
                dirt.lifetime_seconds = 3.2f;
                dirt.start_size = 0.13f; dirt.end_size = 0.09f; dirt.size_jitter = 0.8f;
                dirt.start_color = Vec3(0.42f, 0.31f, 0.2f);
                dirt.end_color = Vec3(0.2f, 0.15f, 0.1f);
                dirt.seed = 0x6a17d17du;
                rt->addEmitter(dirt);

                rt->physicsSettings().grid_density_deposit = 3.0f;
                rt->physicsSettings().grid_temperature_deposit = 5.0f;
                rt->physicsSettings().grid_fuel_deposit = 2.2f;

                RayTrophiSim::SimulationGridDomainDesc dom;
                dom.name = "Ground Burst Gas";
                dom.backend = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                dom.boundary_mode = RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                dom.gas_maccormack_advection = true;
                // Wide and shallow: a ground burst spreads before it climbs.
                dom.bounds_min = Vec3(-4.0f, 0.0f, -4.0f);
                dom.bounds_max = Vec3(4.0f, 6.0f, 4.0f);
                dom.voxel_size = 0.095f;   // 8x6x8 m -> ~84x63x84
                dom.channels |= static_cast<uint32_t>(
                    RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                dom.fire_enabled = true;
                dom.ignition_temperature = 0.12f;
                dom.burn_rate = 6.0f;
                dom.heat_release = 4.2f;
                dom.smoke_generation = 1.6f;      // dirty, sooty ground blast
                dom.flame_dissipation = 2.2f;
                dom.fire_expansion = 0.65f;
                dom.gas_buoyancy_heat = 0.5f;
                dom.gas_buoyancy_density = 0.05f; // heavier, dirt-laden smoke
                dom.gas_vorticity = 0.78f;
                dom.turbulence_strength = 0.78f;
                dom.turbulence_scale = 2.2f;
                dom.turbulence_octaves = 4;
                dom.turbulence_persistence = 0.55f;
                dom.turbulence_speed = 1.2f;
                dom.shader = VolumeShader::createExplosionPreset();
                rt->addGridDomain(dom);

                // Shallow, wide fuel disc right at the ground.
                RayTrophiSim::SimulationFlowSourceDesc blast;
                blast.name = "Ground Fuel Pulse";
                blast.domain_index = 0;
                blast.position = Vec3(0.0f, 0.12f, 0.0f);
                blast.radius = 0.9f;
                blast.velocity = Vec3(0.0f, 1.2f, 0.0f);
                blast.density = 2.4f;
                blast.temperature = 4.5f;
                blast.fuel = 7.0f;
                blast.falloff = 0.5f;
                blast.use_time_limit = true;
                blast.start_time = 0.0f;
                blast.end_time = 0.1f;
                rt->addFlowSource(blast);

                sys.blend_mode = ParticleBlendMode::Additive;
                sys.render.render_in_raytrace = true;
                sys.render.shape = ParticleRenderShape::Tetra;
                sys.render.emissive = true;
                sys.render.base_color = Vec3(1.0f, 0.72f, 0.3f);
                sys.render.emission_strength = 5.0f;
                break;
            }
            case ParticleSystemPreset::Fireball: {
                // Fuel-rich deflagration: little shrapnel, a long fuel burn and
                // strong thermal lift, so the mass rolls upward into a mushroom
                // instead of punching outward. The tall domain is the point.
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Spark);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Realtime);
                rt->setGravity(Vec3(0.0f, -3.2f, 0.0f));   // embers loft
                rt->setLinearDrag(0.55f);

                RayTrophiSim::ParticleEmitterDesc embers;
                embers.name = "Fireball Embers";
                embers.point = Vec3(0.0f, 0.6f, 0.0f);
                embers.direction = Vec3(0.0f, 1.0f, 0.0f);
                embers.rate_per_second = 0.0f;
                embers.burst_count = 180;
                embers.speed = 3.0f;
                embers.spread = 2.4f;
                embers.lifetime_seconds = 3.0f;
                embers.start_size = 0.11f; embers.end_size = 0.02f; embers.size_jitter = 0.55f;
                embers.start_opacity = 1.0f; embers.end_opacity = 0.0f;
                embers.start_color = Vec3(1.0f, 0.88f, 0.5f);
                embers.end_color = Vec3(0.8f, 0.12f, 0.02f);
                embers.angular_velocity = 2.0f; embers.angular_jitter = 4.0f;
                rt->addEmitter(embers);

                // Embers are the fuel carriers here: they keep re-igniting the
                // rising column, which is what sustains a mushroom cap.
                rt->physicsSettings().grid_density_deposit = 1.6f;
                rt->physicsSettings().grid_temperature_deposit = 7.0f;
                rt->physicsSettings().grid_fuel_deposit = 3.5f;

                RayTrophiSim::SimulationGridDomainDesc dom;
                dom.name = "Fireball Gas";
                dom.backend = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                dom.boundary_mode = RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                dom.gas_maccormack_advection = true;
                dom.bounds_min = Vec3(-2.5f, 0.0f, -2.5f);
                dom.bounds_max = Vec3(2.5f, 9.0f, 2.5f);   // tall: room to climb
                dom.voxel_size = 0.085f;   // 5x9x5 m -> ~59x106x59
                dom.channels |= static_cast<uint32_t>(
                    RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                dom.fire_enabled = true;
                dom.ignition_temperature = 0.15f;
                dom.burn_rate = 3.2f;             // slower burn = longer flame life
                dom.heat_release = 4.5f;
                dom.smoke_generation = 1.3f;
                dom.flame_dissipation = 1.2f;     // flame lingers
                dom.fire_expansion = 0.35f;       // sustained roll without late pressure growth
                dom.gas_buoyancy_heat = 1.8f;     // strong lift -> mushroom
                dom.gas_buoyancy_density = 0.03f;
                dom.gas_vorticity = 0.68f;        // curls the cap without injecting runaway energy
                dom.turbulence_strength = 0.58f;
                dom.turbulence_scale = 1.8f;
                dom.turbulence_octaves = 4;
                dom.turbulence_persistence = 0.54f;
                dom.turbulence_speed = 0.9f;
                dom.shader = VolumeShader::createExplosionPreset();
                rt->addGridDomain(dom);

                RayTrophiSim::SimulationFlowSourceDesc fuel;
                fuel.name = "Fireball Fuel Charge";
                fuel.domain_index = 0;
                fuel.position = Vec3(0.0f, 0.6f, 0.0f);
                fuel.radius = 0.8f;
                fuel.velocity = Vec3(0.0f, 2.0f, 0.0f);
                fuel.density = 1.4f;
                fuel.temperature = 4.0f;
                fuel.fuel = 10.0f;
                fuel.falloff = 0.8f;
                fuel.use_time_limit = true;
                fuel.start_time = 0.0f;
                fuel.end_time = 0.35f;            // long charge -> sustained roll
                rt->addFlowSource(fuel);

                sys.blend_mode = ParticleBlendMode::Additive;
                sys.render.render_in_raytrace = true;
                sys.render.shape = ParticleRenderShape::Sphere;
                sys.render.emissive = true;
                sys.render.base_color = Vec3(1.0f, 0.82f, 0.42f);
                sys.render.emission_strength = 9.0f;
                break;
            }
            case ParticleSystemPreset::Flamethrower: {
                // Directional, fuel-rich jet. Low expansion keeps a coherent
                // flame tongue while high source velocity carries ignition to
                // collider surfaces several metres away.
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Spark);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Realtime);
                rt->setGravity(Vec3(0.0f, -2.0f, 0.0f));
                rt->setLinearDrag(0.35f);

                RayTrophiSim::ParticleEmitterDesc sparks;
                sparks.name = "Flamethrower Embers";
                sparks.point = Vec3(0.0f, 1.0f, 0.0f);
                sparks.direction = Vec3(1.0f, 0.05f, 0.0f);
                sparks.rate_per_second = 180.0f;
                sparks.speed = 11.0f;
                sparks.spread = 0.16f;
                sparks.lifetime_seconds = 1.1f;
                sparks.start_size = 0.055f; sparks.end_size = 0.012f;
                sparks.start_opacity = 1.0f; sparks.end_opacity = 0.0f;
                sparks.start_color = Vec3(1.0f, 0.92f, 0.48f);
                sparks.end_color = Vec3(1.0f, 0.12f, 0.01f);
                sparks.seed = 0xf1a6e701u;
                rt->addEmitter(sparks);
                rt->physicsSettings().grid_density_deposit = 0.35f;
                rt->physicsSettings().grid_temperature_deposit = 2.2f;
                rt->physicsSettings().grid_fuel_deposit = 0.75f;

                RayTrophiSim::SimulationGridDomainDesc dom;
                dom.name = "Flamethrower Gas";
                dom.backend = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                dom.boundary_mode = RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                dom.gas_maccormack_advection = true;
                dom.bounds_min = Vec3(-1.0f, -1.0f, -2.2f);
                dom.bounds_max = Vec3(9.0f, 3.5f, 2.2f);
                dom.voxel_size = 0.075f;
                dom.channels |= static_cast<uint32_t>(
                    RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                dom.fire_enabled = true;
                dom.ignition_temperature = 0.18f;
                dom.burn_rate = 4.8f;
                dom.heat_release = 3.8f;
                dom.smoke_generation = 0.48f;
                dom.flame_dissipation = 2.0f;
                dom.fire_expansion = 0.08f;
                dom.gas_buoyancy_heat = 0.48f;
                dom.gas_buoyancy_density = 0.015f;
                dom.gas_vorticity = 0.46f;
                dom.turbulence_strength = 0.62f;
                dom.turbulence_scale = 3.1f;
                dom.turbulence_octaves = 4;
                dom.turbulence_persistence = 0.50f;
                dom.turbulence_speed = 1.6f;
                dom.shader = VolumeShader::createFirePreset();
                // A flamethrower is a hot, optically thin gas jet, not a dense
                // liquid sheet.  Keep this look local to the preset: large
                // density/black absorption values collapse the mean free path
                // and make every fuel-bearing voxel glow as an opaque ribbon.
                dom.shader->name = "Flamethrower Fire";
                dom.shader->density.multiplier = 1.65f;
                dom.shader->density.cutoff_threshold = 0.018f;
                dom.shader->density.edge_falloff = 0.08f;
                dom.shader->scattering.color = Vec3(1.0f, 0.72f, 0.38f);
                dom.shader->scattering.coefficient = 0.12f;
                dom.shader->scattering.anisotropy = 0.18f;
                dom.shader->scattering.multi_scatter = 0.08f;
                dom.shader->absorption.color = Vec3(0.16f, 0.055f, 0.018f);
                dom.shader->absorption.coefficient = 0.55f;
                dom.shader->emission.blackbody_intensity = 9.0f;
                dom.shader->emission.temperature_min = 850.0f;
                dom.shader->emission.temperature_max = 1900.0f;
                dom.shader->emission.color_ramp.enabled = true;
                dom.shader->emission.color_ramp.stops = {
                    {0.00f, Vec3(0.0f, 0.0f, 0.0f), 0.0f},
                    {0.12f, Vec3(0.10f, 0.015f, 0.002f), 0.12f},
                    {0.34f, Vec3(0.90f, 0.12f, 0.008f), 0.58f},
                    {0.62f, Vec3(1.00f, 0.52f, 0.055f), 0.86f},
                    {0.84f, Vec3(1.00f, 0.88f, 0.48f), 0.96f},
                    {1.00f, Vec3(0.82f, 0.91f, 1.00f), 1.0f}
                };
                rt->addGridDomain(dom);

                RayTrophiSim::SimulationFlowSourceDesc jet;
                jet.name = "Flamethrower Fuel Jet";
                jet.domain_index = 0;
                jet.position = Vec3(0.0f, 1.0f, 0.0f);
                jet.radius = 0.24f;
                jet.velocity = Vec3(12.0f, 0.4f, 0.0f);
                jet.velocity_coupling = 16.0f;
                jet.density = 0.42f;
                jet.temperature = 2.8f;
                jet.fuel = 3.6f;
                jet.falloff = 1.8f;
                jet.use_time_limit = false;
                rt->addFlowSource(jet);

                sys.blend_mode = ParticleBlendMode::Additive;
                sys.render.render_in_raytrace = true;
                sys.render.shape = ParticleRenderShape::Sphere;
                sys.render.emissive = true;
                sys.render.base_color = Vec3(1.0f, 0.82f, 0.32f);
                sys.render.emission_strength = 7.0f;
                break;
            }
            case ParticleSystemPreset::BurningFuelSpill: {
                rt->applyPhysicsModePreset(
                    RayTrophiSim::ParticlePhysicsMode::Fluid);
                rt->applyQualityModePreset(
                    RayTrophiSim::ParticleQualityMode::Preview);

                RayTrophiSim::SimulationGridDomainDesc liquid;
                liquid.name="Burning Fuel Liquid";
                liquid.type=RayTrophiSim::SimulationDomainType::Fluid;
                liquid.backend=
                    RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                liquid.boundary_mode=
                    RayTrophiSim::SimulationGridDomainBoundaryMode::Closed;
                liquid.bounds_min=Vec3(-2.5f,0.0f,-2.5f);
                liquid.bounds_max=Vec3(2.5f,1.8f,2.5f);
                liquid.voxel_size=0.10f;
                liquid.fluid_params.applyPreset(
                    RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Oil);
                liquid.fluid_render_mode=
                    RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
                liquid.fluid_seed_min=Vec3(-1.8f,0.15f,-1.8f);
                liquid.fluid_seed_max=Vec3(1.8f,0.55f,1.8f);
                liquid.fluid_seed_particles_per_cell=6;
                liquid.fluid_replace_on_seed=true;
                liquid.fluid_reseed_on_reset=true;
                liquid.fluid_pending_seed=true;
                liquid.fluid_flammable=true;
                liquid.fluid_auto_ignite=true;
                liquid.fluid_ignition_temperature=0.65f;
                liquid.fluid_evaporation_rate=0.45f;
                liquid.fluid_surface_fuel_capacity=5.0f;
                liquid.fluid_combustion_heat_release=2.4f;
                liquid.fluid_combustion_smoke_yield=0.55f;
                liquid.fluid_surface_cooling=0.30f;
                rt->addGridDomain(liquid);

                RayTrophiSim::SimulationGridDomainDesc gas;
                gas.name="Burning Fuel Gas";
                gas.type=RayTrophiSim::SimulationDomainType::Gas;
                gas.backend=
                    RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                gas.boundary_mode=
                    RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                gas.bounds_min=Vec3(-2.5f,0.0f,-2.5f);
                gas.bounds_max=Vec3(2.5f,5.0f,2.5f);
                gas.voxel_size=0.10f;
                gas.channels|=static_cast<uint32_t>(
                    RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                gas.fire_enabled=true;
                gas.ignition_temperature=0.30f;
                gas.burn_rate=1.35f;
                gas.heat_release=2.2f;
                gas.smoke_generation=0.65f;
                gas.flame_dissipation=2.6f;
                gas.gas_buoyancy_heat=1.15f;
                gas.gas_buoyancy_density=0.06f;
                gas.gas_vorticity=0.42f;
                gas.fire_expansion=0.12f;
                gas.turbulence_strength=0.28f;
                gas.turbulence_scale=1.35f;
                gas.turbulence_octaves=4;
                gas.shader=VolumeShader::createFirePreset();
                gas.shader->name="Burning Fuel Fire";
                rt->addGridDomain(gas);

                // The coupled domains provide the render geometry/volume.
                // This preset needs no decorative discrete-particle layer.
                sys.render.render_in_raytrace=false;
                break;
            }
            case ParticleSystemPreset::IgnitedFuelJet: {
                rt->applyPhysicsModePreset(
                    RayTrophiSim::ParticlePhysicsMode::Fluid);
                rt->applyQualityModePreset(
                    RayTrophiSim::ParticleQualityMode::Preview);

                const int liquid_index =
                    static_cast<int>(rt->gridDomains().size());
                const std::string suffix = " #" + std::to_string(sys.id);

                RayTrophiSim::SimulationGridDomainDesc liquid;
                liquid.name = "Ignited Fuel Jet Liquid" + suffix;
                liquid.type = RayTrophiSim::SimulationDomainType::Fluid;
                liquid.backend =
                    RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                liquid.boundary_mode =
                    RayTrophiSim::SimulationGridDomainBoundaryMode::Closed;
                liquid.bounds_min = Vec3(-1.0f, 0.0f, -2.0f);
                liquid.bounds_max = Vec3(7.0f, 2.0f, 2.0f);
                liquid.voxel_size = 0.10f;
                liquid.resource_budget_mb = 768;
                liquid.fluid_params.applyPreset(
                    RayTrophiSim::Fluid::APICSolverParams::FluidPreset::Oil);
                // Thinner than the Oil preset: a spilled fuel puddle spreads. The
                // old 0.16 was on the unitless dial (Oil sat at 3.0 there), i.e.
                // "much thinner than oil" — restated in m²/s against the new
                // Oil ν of 1e-4.
                liquid.fluid_params.kinematic_viscosity = 5.0e-6f;
                liquid.fluid_max_particles = 80000;
                liquid.fluid_render_mode =
                    RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
                liquid.fluid_pending_seed = false;
                liquid.fluid_reseed_on_reset = false;
                liquid.fluid_replace_on_seed = false;
                liquid.fluid_surface_ior = 1.44f;
                liquid.fluid_surface_roughness = 0.075f;
                liquid.fluid_surface_foam = 0.0f;
                liquid.fluid_foam_params.enabled = false;
                liquid.fluid_flammable = true;
                // The pilot is the ignition source for this preset.  Keep
                // auto-ignite off, but use the same low normalized threshold
                // as the gas pilot so a short-lived pilot can actually hand
                // heat back to the liquid surface.
                liquid.fluid_auto_ignite = false;
                liquid.fluid_ignition_temperature = 0.35f;
                liquid.fluid_evaporation_rate = 0.28f;
                liquid.fluid_surface_fuel_capacity = 4.5f;
                liquid.fluid_combustion_heat_release = 2.25f;
                liquid.fluid_combustion_smoke_yield = 0.48f;
                liquid.fluid_surface_cooling = 0.24f;
                liquid.shader = std::make_shared<VolumeShader>();
                liquid.shader->name = "Amber Fuel Surface" + suffix;
                liquid.shader->density.multiplier = 1.0f;
                liquid.shader->density.cutoff_threshold = 0.01f;
                liquid.shader->scattering.color =
                    Vec3(0.96f, 0.82f, 0.56f);
                liquid.shader->scattering.coefficient = 0.0f;
                liquid.shader->absorption.color =
                    Vec3(0.10f, 0.42f, 0.92f);
                liquid.shader->absorption.coefficient = 0.42f;
                rt->addGridDomain(liquid);

                const int gas_index =
                    static_cast<int>(rt->gridDomains().size());
                RayTrophiSim::SimulationGridDomainDesc gas;
                gas.name = "Ignited Fuel Jet Gas" + suffix;
                gas.type = RayTrophiSim::SimulationDomainType::Gas;
                gas.backend =
                    RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                gas.boundary_mode =
                    RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                gas.bounds_min = Vec3(-1.0f, 0.0f, -2.0f);
                gas.bounds_max = Vec3(7.0f, 5.5f, 2.0f);
                gas.voxel_size = 0.10f;
                gas.resource_budget_mb = 768;
                gas.channels |= static_cast<uint32_t>(
                    RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                gas.gas_maccormack_advection = true;
                gas.fire_enabled = true;
                gas.ignition_temperature = 0.32f;
                gas.burn_rate = 1.75f;
                gas.heat_release = 2.45f;
                gas.smoke_generation = 0.52f;
                gas.flame_dissipation = 2.35f;
                gas.fire_max_temperature = 8.0f;
                gas.fire_expansion = 0.10f;
                gas.gas_buoyancy_heat = 0.95f;
                gas.gas_buoyancy_density = 0.045f;
                gas.gas_vorticity = 0.38f;
                gas.turbulence_strength = 0.30f;
                gas.turbulence_scale = 1.45f;
                gas.turbulence_octaves = 4;
                gas.turbulence_persistence = 0.52f;
                gas.turbulence_speed = 0.85f;
                gas.shader = VolumeShader::createFirePreset();
                gas.shader->name = "Ignited Fuel Jet Fire" + suffix;
                gas.shader->density.multiplier = 1.55f;
                gas.shader->density.cutoff_threshold = 0.012f;
                gas.shader->density.edge_falloff = 0.06f;
                gas.shader->scattering.color =
                    Vec3(0.86f, 0.78f, 0.68f);
                gas.shader->scattering.coefficient = 0.16f;
                gas.shader->scattering.anisotropy = 0.28f;
                gas.shader->scattering.multi_scatter = 0.16f;
                gas.shader->absorption.color =
                    Vec3(0.32f, 0.24f, 0.18f);
                gas.shader->absorption.coefficient = 0.46f;
                gas.shader->emission.blackbody_intensity = 5.8f;
                gas.shader->emission.temperature_min = 720.0f;
                gas.shader->emission.temperature_max = 1850.0f;
                rt->addGridDomain(gas);

                RayTrophiSim::SimulationFlowSourceDesc liquid_jet;
                liquid_jet.name = "Fuel Nozzle" + suffix;
                liquid_jet.domain_index = liquid_index;
                liquid_jet.source_mode =
                    RayTrophiSim::SimulationFlowSourceMode::Point;
                liquid_jet.position = Vec3(0.0f, 1.45f, 0.0f);
                liquid_jet.radius = 0.20f;
                liquid_jet.velocity = Vec3(3.4f, -1.15f, 0.0f);
                liquid_jet.fluid_particles_per_second = 5200.0f;
                liquid_jet.fluid_velocity_spread = 0.12f;
                liquid_jet.use_time_limit = true;
                liquid_jet.start_time = 0.0f;
                liquid_jet.end_time = 8.0f;
                liquid_jet.use_particle_limit = true;
                liquid_jet.max_emitted_particles = 42000;
                rt->addFlowSource(liquid_jet);

                // A short pilot ignites the pool after it has had time to reach
                // the floor. It is not a permanent decorative flame: afterward
                // the finite liquid surface fuel owns the combustion.
                RayTrophiSim::SimulationFlowSourceDesc pilot;
                pilot.name = "Fuel Jet Pilot" + suffix;
                pilot.domain_index = gas_index;
                pilot.position = Vec3(1.35f, 0.28f, 0.0f);
                // Cover the nozzle/floor contact band.  A narrow pilot could
                // heat a gas cell beside the exposed liquid surface while
                // never touching the surface cell sampled by the coupling.
                pilot.radius = 0.65f;
                pilot.velocity = Vec3(0.0f, 0.85f, 0.0f);
                pilot.velocity_coupling = 12.0f;
                pilot.density = 0.16f;
                pilot.temperature = 8.0f;
                pilot.fuel = 2.2f;
                pilot.falloff = 1.35f;
                pilot.use_time_limit = true;
                pilot.start_time = 0.45f;
                pilot.end_time = 4.0f;
                rt->addFlowSource(pilot);

                // The coupled SurfaceSDF and gas domain are the final render.
                sys.render.render_in_raytrace = false;
                break;
            }
            // ── Nuclear detonation ───────────────────────────────────────────
            //
            // ★ Two presets, ONE recipe at two scales, and the scale is a single
            // factor every length below is written against. Two hand-tuned
            // copies would drift apart the first time either was calibrated.
            //
            // They stay SEPARATE presets rather than one with a size dial
            // because they are not the same shot at two sizes: the cinematic one
            // is tuned to be iterated on (metre-scale box, viewport-affordable
            // cell count, a few seconds of sim), the physical one is a
            // kilometre-scale offline domain. A shared dial would give every
            // parameter here a hidden "which scale am I in" meaning.
            //
            // ★ What makes this a mushroom rather than a big fire:
            //   1. `gas_ambient_stratification` gives the plume a ceiling of its
            //      OWN (h* ≈ heat anomaly / stratification). Without it the cap
            //      is shaped by the domain lid — it silently changes when the
            //      box is resized, and looks exactly like a settled cloud.
            //   2. High vorticity confinement rolls the rising cap into a torus.
            //      That roll is what reads as "mushroom" instead of "column".
            //   3. A LATE ground-level dust source. The stem is not blast
            //      debris; it is the afterwind — air drawn back in behind the
            //      departing fireball, lifting dust seconds later. Authored as a
            //      time-windowed flow source that starts AFTER the fireball has
            //      cleared the ground.
            //
            // The burn is short and violent (high burn_rate, fast flame
            // dissipation) unlike Fireball's slow deflagration: a weapon's light
            // is over in a fraction of a second and everything after it is hot
            // dust. Copying Fireball's long fuel burn is the obvious mistake and
            // it produces a petrol fireball wearing a mushroom's shape.
            case ParticleSystemPreset::NuclearCinematic:
            case ParticleSystemPreset::NuclearPhysical: {
                const bool physical = (preset == ParticleSystemPreset::NuclearPhysical);
                // Length scale, in metres per cinematic unit.
                const float S = physical ? 80.0f : 1.0f;
                // ★ Time does NOT scale with S. The solver's seconds are the
                // timeline's seconds, and a shot nobody can sit through is not a
                // better shot; the physical preset stretches the stages only far
                // enough for them to read at its size.
                const float T = physical ? 6.0f : 1.0f;

                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Spark);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Realtime);
                rt->setGravity(Vec3(0.0f, -9.81f, 0.0f));
                rt->setLinearDrag(0.8f);

                // Blast debris. These are NOT the stem: they arc and fall back.
                // Kept sparse on the physical preset, where a single fragment is
                // sub-voxel and reads as noise rather than as debris.
                RayTrophiSim::ParticleEmitterDesc debris;
                debris.name = "Detonation Debris";
                // ** RE-DERIVED 2026-09-21. Spawning AT the origin rather than
                // a quarter-unit up: the fireball's own source sits at 1.2, so
                // lifting the debris only pushed it into the flash where it was
                // invisible. A direction shorter than unit length biases the
                // cone downward without narrowing the spread, which is what
                // makes the fragments arc instead of fountaining.
                debris.point = Vec3(0.0f, 0.0f, 0.0f);
                debris.direction = Vec3(0.0f, 0.7f, 0.0f);
                debris.rate_per_second = 0.0f;
                debris.burst_count = physical ? 120 : 320;
                debris.speed = 11.9f * (physical ? 6.0f : 1.0f);
                debris.spread = 2.6f;
                // Long enough to still be falling while the stem forms; at
                // 3.5 the arcs were gone before the cap had shape.
                debris.lifetime_seconds = 5.85f * T;
                // ** Was left at the 1.0 default, which is NOT neutral: drag is
                // 0.8 and gravity 9.81, so a heavier fragment flattened its arc.
                debris.mass = 0.4f;
                debris.start_size = 0.09f * S; debris.end_size = 0.02f * S;
                debris.size_jitter = 0.6f;
                debris.start_opacity = 1.0f; debris.end_opacity = 0.0f;
                debris.start_color = Vec3(1.0f, 0.93f, 0.62f);
                debris.end_color = Vec3(0.35f, 0.18f, 0.10f);
                debris.angular_velocity = 3.0f; debris.angular_jitter = 5.0f;
                debris.seed = 0x4e554b45u;
                rt->addEmitter(debris);

                // ★ Debris carries dust and heat, NOT fuel. The fuel is spent in
                // the first flash; depositing more along the debris arcs would
                // keep re-igniting the column and turn the stem back into a fire
                // plume — which is exactly what the Fireball preset wants and
                // this one must not have.
                rt->physicsSettings().grid_density_deposit = 2.2f;
                rt->physicsSettings().grid_temperature_deposit = 3.0f;
                rt->physicsSettings().grid_fuel_deposit = 0.0f;

                RayTrophiSim::SimulationGridDomainDesc dom;
                dom.name = physical ? "Nuclear Gas (Physical)" : "Nuclear Gas";
                dom.backend = RayTrophiSim::SimulationDomainBackend::GPU_Vulkan;
                dom.boundary_mode = RayTrophiSim::SimulationGridDomainBoundaryMode::Open;
                dom.gas_maccormack_advection = true;
                dom.bounds_min = Vec3(-11.0f * S, 0.0f, -11.0f * S);
                dom.bounds_max = Vec3( 11.0f * S, 34.0f * S,  11.0f * S);
                // Cinematic: 22x34x22 m at 0.22 -> 100x155x100 (~1.5M cells).
                // Physical: 1760x2720x1760 m at 17.6 m -> the SAME cell count.
                // ★ The physical preset is not FINER, it is BIGGER. Holding the
                // cell count fixed is what makes the two comparable: if the cap
                // settles at a different fraction of the domain height, that is
                // the scale talking and not the resolution.
                // ** RE-DERIVED 2026-09-21 from a hand-tuned scene. 0.17 gives
                // 130x200x130 (3.4M cells) at cinematic scale - finer than the
                // 0.22 this shipped with, and the cap holds its torus at it.
                dom.voxel_size = 0.17f * S;
                dom.quality_profile = physical
                    ? RayTrophiSim::SimulationDomainQualityProfile::Final
                    : RayTrophiSim::SimulationDomainQualityProfile::Preview;
                dom.resource_budget_mb = physical ? 4096u : 1536u;
                dom.channels |= static_cast<uint32_t>(
                    RayTrophiSim::SimulationGridDomainChannelFlags::Fuel);
                dom.fire_enabled = true;
                dom.ignition_temperature = 0.12f;
                dom.burn_rate = 9.0f;            // weapon, not deflagration
                dom.heat_release = 7.0f;
                dom.smoke_generation = 2.6f;     // the cloud is mostly this
                dom.flame_dissipation = 3.4f;    // the flash is brief
                dom.fire_max_temperature = 10.0f;
                dom.fire_expansion = 1.15f;      // the shock; drives the ground ring
                dom.gas_buoyancy_heat = 2.6f;
                dom.gas_buoyancy_density = 0.02f;
                // ── Ground dust, lifted by the blast's own wind. ★ This REPLACES
                // a "Ground Dust Skirt" flow source that this preset shipped
                // with: a ring at the origin whose radius the author typed, which
                // did not follow the shock and looked the same on every frame.
                // Now the shock scours the floor it crosses and the skirt expands
                // with the front, because that is what actually happens.
                //
                // ★ The afterwind stage below ALSO lifts its own dust through this
                // rule, which is why its flow source no longer carries any: a
                // mushroom's stem is ground material pulled up behind the
                // fireball, not smoke injected at the base.
                dom.gas_surface_dust_enabled = true;
                // Lower than the 6 m/s default: the afterwind's ground-level
                // inflow is gentler than the shock, and it has to clear the bar
                // too or there is no stem at all.
                dom.gas_surface_dust_threshold = 4.0f;
                dom.gas_surface_dust_emission = 0.6f;
                dom.gas_surface_dust_max_density = 2.5f;
                // ★ MEASURED 2026-09-20: with an unlimited reservoir this preset
                // produced a uniform ground SHEET (widest slice pinned at 0.07 m,
                // growing 4 -> 10 m) and a stem as thick as the cap. A finite
                // reserve is what turns it back into a front that passes.
                dom.gas_surface_dust_supply = 3.0f;

                // ── Field loss. ★ MEASURED 2026-09-20: without this override the
                // cloud is gone by ten seconds. The solver's global rates are
                // 0.5/s for smoke and heat alike, chosen for the thin smoke a
                // spark carries, and a hybrid preset runs in Spark mode so it
                // inherits them: peak heat 9.8 -> 0.04 and 311k active cells ->
                // 8.9k, with every authored parameter here correct.
                dom.gas_dissipation_override = true;
                // ***** MEASURED 2026-09-21: 0.012f WAS EFFECTIVELY ZERO.
                // Over the 8.3 s shot it removes 9% of the smoke. Every flow
                // source is finished by 4.5 s (frame 108), so after that nothing
                // produces and nothing removes: the cloud simply sat there and
                // spread until it filled the box. Active cells and fill never
                // reached a steady state - 0.25 -> 0.48 -> 0.64 -> 0.77 - and a
                // full domain has only one free direction left, which is why
                // this read as the cap "collapsing" long after the buoyancy term
                // had been fixed.
                //
                // At 0.18 removal balances spreading: cells peak at frame 110
                // and settle back (1.29M -> 1.19M, fill 0.35 flat). That plateau
                // is what reads on screen as smoke that thins as it climbs.
                //
                // * Pulverised ground still does not evaporate - this is not
                // evaporation. It is the sub-grid dispersal that a 0.17 m voxel
                // cannot represent: real smoke keeps mixing into clear air long
                // after it stops being resolvable.
                //
                // * Bonus, and it falls out for free: as the smoke thins its
                // `presence` drops, so the stratification term weakens and the
                // thinning cloud recovers lift. Buoyancy and dissipation read
                // the same density, so the two stay consistent by construction.
                dom.gas_density_dissipation = 0.18f;
                // The cloud SHOULD cool -- that is what settles it -- just not
                // 20x faster than it rises.
                dom.gas_temperature_dissipation = 0.22f;
                // Fuel is spent in the flash and must not linger.
                dom.gas_fuel_dissipation = 0.5f;
                // ★★★ THE PARAMETER THIS PRESET EXISTS FOR: the plume gets a
                // ceiling of its OWN rather than one shaped by the domain lid,
                // so the cap does not silently change when the box is resized.
                //
                // ★★ STILL COUPLED TO gas_temperature_dissipation ABOVE - a plume
                // that cools slower keeps its lift longer and settles higher - but
                // the coupling is no longer dangerous. The term is one-sided now,
                // so getting this wrong costs cap ALTITUDE and can no longer turn
                // the cap around and drive it into the ground.
                //
                // ** RE-DERIVED 2026-09-21 against the ONE-SIDED term.
                // The old 6.5/(26*S) = 0.25/S encoded "an anomaly of 6.5 balances
                // at 26 m" - but the anomaly does not stay at 6.5, it decays at
                // 0.22/s, so the ceiling it computed collapsed with it. The
                // environment term can no longer push down (see buoyantAnomaly in
                // GridFluidSolver.cpp), so this dial now only decides how early
                // the lift is cancelled, and it wants to be much smaller.
                //
                // MEASURED: at 0.05/S the cap rises and parks; at 0.25/S it parks
                // lower but still reaches the lid on momentum; at 1.2/S the rise
                // is suppressed so hard the gas accumulates in place and the
                // domain's own box edges become visible in frame.
                //
                // Stratification is heat per world unit of HEIGHT, so it scales
                // as 1/S - the same inverse-length rule as turbulence_scale.
                dom.gas_ambient_stratification = 0.05f / S;
                dom.gas_vorticity = 1.15f;       // rolls the cap into a torus
                dom.turbulence_strength = 0.72f;
                dom.turbulence_scale = 1.6f / S; // spatial FREQUENCY: inverse length
                // ** ASKING FOR 8 IS NOT GETTING 8. effectiveTurbulenceOctaves
                // clamps to what the voxel can actually advect (4 cells per
                // wavelength), and at 0.17 m with scale 1.6 that is THREE. The
                // extra five are requested so the same preset resolves more
                // detail when the domain is refined, and are discarded until
                // then - gas.get_settings reports turbulence_octaves_effective,
                // which is the number to trust.
                dom.turbulence_octaves = 8;
                // ★ MEASURED 2026-09-20: at 0.56 with lacunarity 2 the octave
                // amplitudes RISE (1.12^o), so octave 4 — wavelength 0.245 m
                // against a 0.1375 m voxel — was the STRONGEST component. That is
                // grid-scale speckle, and it is what made the cap read as
                // chewed-up lumps instead of a rolling torus.
                dom.turbulence_persistence = 0.40f;
                dom.turbulence_speed = 0.8f;
                dom.shader = VolumeShader::createExplosionPreset();
                dom.shader_preset = "fire";
                if (dom.shader) {
                    // A weapon's core is white, not orange. Raising
                    // temperature_max is what stops it clipping to a flat orange
                    // disc at the top of the blackbody ramp.
                    // ** RE-DERIVED 2026-09-21 from a hand-tuned scene. The
                    // flash reads white without blowing the whole cap out: the
                    // window starts well above ambient so cooled smoke stops
                    // glowing, and the intensity is a quarter of what this
                    // shipped with because density_multiplier below more than
                    // doubled.
                    dom.shader->emission.blackbody_intensity = 16.667f;
                    dom.shader->emission.temperature_min = 1340.0f;
                    // ***** 5000 WAS BELOW THE SCENE'S OWN RANGE.
                    // Gas temperature reaches the shader as solver heat x3000, so
                    // this scene spans 5475-19769 K - every cell in the cap sat
                    // ABOVE the clamp and rendered at one flat maximum, which is
                    // what made the cap a white slab with no internal shading.
                    //
                    // Since the T^4 radiance term landed this value has a second
                    // job: it is also what the radiance is normalised against, so
                    // putting it below the scene's peak pins the whole cloud at
                    // radiance 1.0 and throws the fix away. It must sit at or
                    // above peak_temperature x 3000 - gas.measure_plume reports
                    // that directly.
                    dom.shader->emission.temperature_max = 20000.0f;
                    dom.shader->emission.temperature_scale = 1.6f;
                    // Condensation read: this cap is water and pulverised
                    // ground, not soot — bright, weakly absorbing, strongly
                    // forward-scattering.
                    // ★ An APPEARANCE claim, not a phase change. There is no
                    // moisture channel in the gas grid, so nothing here
                    // condenses; it is a look that happens to be the right one.
                    dom.shader->scattering.color = Vec3(0.82f, 0.80f, 0.78f);
                    // ** The big one: sigma_s 5.5 -> 0.39 and sigma_a 0.9 ->
                    // 0.39, with density_multiplier carrying the opacity instead.
                    // At 5.5 a single voxel was effectively opaque, so the march
                    // terminated a step or two in and the cap rendered as a hard
                    // shell - no depth, and the internal rolls invisible.
                    dom.shader->scattering.coefficient = 0.84f;
                    dom.shader->scattering.anisotropy = 0.55f;
                    dom.shader->scattering.multi_scatter = 0.85f;
                    // ** RE-BALANCED 2026-09-21: sigma_a is now 2x sigma_s and
                    // density_multiplier came DOWN. The previous pair made the
                    // cap read as a flat white cut-out; absorbing more per unit
                    // density while carrying less total density is what gives
                    // the lobes their shading and keeps the bright side off the
                    // clip point.
                    dom.shader->absorption.coefficient = 1.72f;
                    dom.shader->density.multiplier = 13.194f;
                    dom.shader->density.cutoff_threshold = 0.007f;
                    dom.shader->absorption.color = Vec3(0.20f, 0.18f, 0.16f);
                }
                rt->addGridDomain(dom);

                // ── Stage 1: the detonation. One frame's worth.
                RayTrophiSim::SimulationFlowSourceDesc core;
                core.name = "Detonation Core";
                core.domain_index = 0;
                core.position = Vec3(0.0f, 1.2f * S, 0.0f);
                // ** RE-DERIVED 2026-09-21: 0.9 was ~5 voxels across at this
                // grid and the flash had no room to form a pressure structure.
                core.radius = 3.02f * S;
                core.velocity = Vec3(0.0f, 0.0f, 0.0f);  // fire_expansion does the work
                core.density = 1.1f;
                core.temperature = 10.0f;
                // ** THE FUEL MOVED HERE, and that is the weapon reading: a
                // device burns its charge in the first flash, it does not feed a
                // plume. Stage 2 dropped from 5.0 to 0.3 to pay for this.
                core.fuel = 87.6f;
                core.falloff = 0.5f;
                core.use_time_limit = true;
                core.start_time = 0.0f;
                core.end_time = 0.05f * T;
                rt->addFlowSource(core);

                // ── Stage 2: the fireball's own burn, already rising.
                RayTrophiSim::SimulationFlowSourceDesc fireball;
                fireball.name = "Fireball Rise";
                fireball.domain_index = 0;
                fireball.position = Vec3(0.0f, 2.2f * S, 0.0f);
                fireball.radius = 1.8f * S;
                fireball.velocity = Vec3(0.0f, 5.0f, 0.0f);
                fireball.velocity_coupling = 6.0f;
                fireball.density = 0.9f;
                fireball.temperature = 6.0f;
                fireball.fuel = 0.0f;   // see the note on core.fuel above
                fireball.falloff = 0.9f;
                fireball.use_time_limit = true;
                fireball.start_time = 0.05f * T;
                fireball.end_time = 0.55f * T;
                rt->addFlowSource(fireball);

                // ── Stage 3: the AFTERWIND — this is the stem, and its TIMING
                // is the whole trick. It starts after the fireball has cleared
                // the ground, so the dust it lifts is drawn up BEHIND the cap
                // into a narrow column instead of joining the blast.
                RayTrophiSim::SimulationFlowSourceDesc stem;
                stem.name = "Stem Afterwind";
                stem.domain_index = 0;
                stem.position = Vec3(0.0f, 0.5f * S, 0.0f);
                stem.radius = 1.95f * S;
                stem.velocity = Vec3(0.0f, 11.0f, 0.0f);
                stem.velocity_coupling = 9.0f;
                // ★★ THIS USED TO BE 0.0, AND THE REASON IT WAS ZERO STILL
                // STANDS - it is being overridden deliberately, not forgotten.
                //
                // The original note: this source is a WIND, not a smoke emitter,
                // and the material it carries up should be dust the surface rule
                // lifted off the ground under it, because that is what a stem is
                // made of. Injecting smoke here makes the stem look right whether
                // or not gas_surface_dust_enabled does anything.
                //
                // ★ SO THE COST IS EXPLICIT: with a non-zero density here, the
                // stem is NO LONGER EVIDENCE that the ground-dust rule works. To
                // test that rule, set this to 0 and check a stem still forms.
                // Test-NuclearPreset.ps1's skirt assertions still cover the
                // ground ring, which is the other half of the same feature.
                stem.density = 2.35f;
                stem.temperature = 0.9f;   // warm enough to rise, too cool to cap
                // ***** BACK TO ZERO, AND THIS ONE IS LOAD-BEARING.
                // At 0.5 the column re-ignited along its whole length and the
                // stem rendered as an orange fire pillar from ground to cap -
                // exactly what the original note on this preset warned: fuel
                // deposited along the column turns the stem back into a fire
                // plume. A real stem is dark lifted dust. MEASURED by eye
                // 2026-09-21: at 0.0 the column goes dark and keeps only the
                // residual blackbody glow near the top, which is correct.
                stem.fuel = 0.0f;
                stem.falloff = 1.1f;
                stem.use_time_limit = true;
                stem.start_time = 0.45f * T;
                stem.end_time = 4.5f * T;
                rt->addFlowSource(stem);

                sys.blend_mode = ParticleBlendMode::Additive;
                sys.render.render_in_raytrace = true;
                sys.render.shape = ParticleRenderShape::Sphere;
                sys.render.emissive = true;
                sys.render.base_color = Vec3(1.0f, 0.86f, 0.58f);
                sys.render.emission_strength = 14.0f;
                break;
            }
        }

        applyParticleSystemEnabledState(sys);
        if (particle_systems.size() != systems_before + 1u) {
            SCENE_LOG_ERROR(
                "[ParticlePreset] Additive preset invariant failed: system count " +
                std::to_string(systems_before) + " -> " +
                std::to_string(particle_systems.size()) + ".");
        } else {
            SCENE_LOG_INFO(
                "[ParticlePreset] Added '" + sys.name + "'; systems=" +
                std::to_string(particle_systems.size()) + ".");
        }
        return sys;
    }
