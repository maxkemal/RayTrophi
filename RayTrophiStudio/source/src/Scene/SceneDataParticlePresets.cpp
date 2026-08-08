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
                sys.render.color_end  = Vec3(1.0f, 0.2f, 0.05f);
                sys.render.color_buckets = 10;
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
                sys.render.color_end  = Vec3(0.25f, 0.06f, 0.02f);
                sys.render.color_buckets = 8;
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
                sys.render.color_end  = Vec3(0.2f, 0.08f, 0.03f);
                sys.render.color_buckets = 8;
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
                sys.render.color_end  = Vec3(0.6f, 0.1f, 0.02f);
                sys.render.color_buckets = 10;
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
                sys.render.color_end = Vec3(0.85f, 0.08f, 0.01f);
                sys.render.color_buckets = 10;
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
                liquid.fluid_params.viscosity = 0.16f;
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
