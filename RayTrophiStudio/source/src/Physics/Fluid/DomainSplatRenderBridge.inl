// Included by ParticleRenderBridge.cpp; shares its private render pool helpers.
void SceneData::syncDomainFluidParticleInstances(bool enable_rt_geometry) {
    if (g_optix_rebuild_in_progress.load() || g_viewport_rebuild_in_progress.load()) {
        return;
    }
    auto& im = InstanceManager::getInstance();

    bool structural_change = false;
    bool motion_change = false;

    for (auto& system : particle_systems) {
        if (!system.runtime) continue;
        const auto& states  = system.runtime->gridDomainStates();
        auto& domains       = system.runtime->gridDomains();

        // Shrink parallel-indexed transient state if domains disappeared.
        // ★ "Disappeared" must mean the DESCRIPTOR is gone, not that the runtime
        //   state vector is momentarily short. resetGridDomainStates() clears and
        //   re-resizes `states`, so keying teardown off it deleted and recreated
        //   this instance group every time the sim reset — visible in the log as
        //   back-to-back "Deleted group '[FluidDomParticles] …'" / "Created group
        //   '[FluidDomParticles] …'". Same defect shape as the domain-volume
        //   teardown in scene_data.h: identity churn under a baked TLAS.
        const std::size_t live_domain_count =
            (std::max)(states.size(), domains.size());
        for (std::size_t d = live_domain_count;
             d < system.domain_particle_render_group_ids.size(); ++d) {
            const int gid = system.domain_particle_render_group_ids[d];
            if (gid >= 0) {
                im.deleteGroup(gid);
                g_fluid_source_state.erase(gid);
                structural_change = true;
            }
        }
        system.domain_particle_render_group_ids.resize(live_domain_count, -1);
        system.domain_particle_pool_capacities.resize(live_domain_count, 0);

        for (std::size_t d = 0; d < states.size(); ++d) {
            const auto& state  = states[d];
            const auto& render_particles = d < domains.size()
                ? RayTrophiSim::Fluid::domainSplatParticles(domains[d], state)
                : state.particles;
            const bool is_fluid = RayTrophiSim::simulationDomainHasLiquid(state);
            // The one view decision (FluidViewResolver); splat vs surface vs fog
            // is never re-derived from the mode or the bindings in this file.
            RayTrophiSim::Fluid::FluidViewPlan view_plan;
            if (is_fluid && d < domains.size()) {
                view_plan = RayTrophiSim::Fluid::resolveFluidViews(
                    domains[d], domains[d].fluid_visual_samples.enabled
                        ? RayTrophiSim::Fluid::distinctViewKeys(render_particles)
                        : RayTrophiSim::Fluid::cachedDistinctViewKeys(render_particles, state.version));
            }

            // Solid, Matcap, Material Preview and RayFusion share the native
            // SurfaceSDF volume allocation. Keep the bounded sphere fallback
            // only until that pipeline has been created successfully, and only
            // for a domain that has a surface to stand in for.
            const bool needs_raster_sdf_proxy =
                g_solid_viewport_active && !g_material_preview_viewport_active &&
                !g_native_surface_sdf_viewport_available && view_plan.surface;

            // Explicit particle/Splat authoring always wins. The compatibility
            // proxy only fills SurfaceSDF in raster modes without a native pass.
            const bool lifecycle_alive =
                system.visible && system.enabled && state.valid && is_fluid &&
                d < domains.size();
            const bool has_explicit_splats = view_plan.splat;
            const bool render_eligible = enable_rt_geometry && lifecycle_alive &&
                (has_explicit_splats || needs_raster_sdf_proxy);

            int& group_id = system.domain_particle_render_group_ids[d];

            if (!lifecycle_alive) {
                if (group_id >= 0) {
                    if (InstanceGroup* g = im.getGroup(group_id)) {
                        if (!g->instances.empty()) structural_change = true;
                    }
                    im.deleteGroup(group_id);
                    g_fluid_source_state.erase(group_id);
                    group_id = -1;
                    system.domain_particle_pool_capacities[d] = 0;
                }
                continue;
            }

            // Preserve explicit splat pools across temporary backend visibility
            // changes. Pure SurfaceSDF releases its compatibility pool below.
            if (!render_eligible) {
                InstanceGroup* existing = (group_id >= 0) ? im.getGroup(group_id) : nullptr;
                // A pure SurfaceSDF consumer has no use for the particle pool.
                // Keeping it hidden retained every CPU transform and every
                // viewport instance allocation after a temporary Splat edit.
                // Delete it here; Solid/Matcap recreates its bounded fallback
                // only if that compatibility preview is requested again.
                if (!has_explicit_splats && !needs_raster_sdf_proxy) {
                    if (existing) {
                        im.deleteGroup(group_id);
                        structural_change = true;
                    }
                    if (group_id >= 0) g_fluid_source_state.erase(group_id);
                    group_id = -1;
                    system.domain_particle_pool_capacities[d] = 0;
                    continue;
                }
                if (!existing) continue;
                if (existing->rendered_rt_excluded != !has_explicit_splats) {
                    existing->rendered_rt_excluded = !has_explicit_splats;
                    structural_change = true;
                }
                bool had_visible = false;
                for (auto& tr : existing->instances) {
                    had_visible |= tr.scale.x != 0.0f || tr.scale.y != 0.0f || tr.scale.z != 0.0f;
                    tr.scale = Vec3(0.0f);
                }
                if (had_visible) {
                    existing->gpu_dirty = true;
                    g_fluid_source_state[group_id].content_hash = 0u;
                    // Raster keeps every pool slot and masks zero-scale ones
                    // (syncRasterInstanceTransforms), so hiding is a motion
                    // update there too -- no raster rebuild is needed.
                    motion_change = true;
                }
                continue;
            }

            // Authored splat view (a label route) with no splat parcel this frame:
            // keep the pool - deleting it is a structural RT rebuild, and it would
            // come back with the next droplet - hide its instances, skip the walk.
            // A pool that never existed is not created until a parcel needs it.
            if (!view_plan.anyLiveIn(RayTrophiSim::Fluid::FluidView::Splat) &&
                !needs_raster_sdf_proxy) {
                InstanceGroup* existing = (group_id >= 0) ? im.getGroup(group_id) : nullptr;
                if (!existing) continue;
                bool had_visible = false;
                for (auto& tr : existing->instances) {
                    had_visible |= tr.scale.x != 0.0f || tr.scale.y != 0.0f || tr.scale.z != 0.0f;
                    tr.scale = Vec3(0.0f);
                }
                if (had_visible) {
                    existing->gpu_dirty = true;
                    g_fluid_source_state[group_id].content_hash = 0u;
                    motion_change = true;
                }
                continue;
            }

            // A rewind commonly lands on a frame before the first molten/APIC
            // particle exists. Keep an already-created RT pool alive and merely
            // hide its instances. Deleting it here and recreating it one frame
            // later forces a full BLAS/TLAS structural rebuild exactly while the
            // melted source mesh and SurfaceSDF bindings are also being restored.
            // That three-way transition was a repeatable Vulkan TDR trigger.
            if (render_particles.empty()) {
                InstanceGroup* existing = (group_id >= 0) ? im.getGroup(group_id) : nullptr;
                if (!existing) continue;
                bool had_visible = false;
                for (auto& tr : existing->instances) {
                    if (tr.scale.x != 0.0f || tr.scale.y != 0.0f || tr.scale.z != 0.0f)
                        had_visible = true;
                    tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                }
                if (had_visible) {
                    existing->gpu_dirty = true;
                    g_fluid_source_state[group_id].content_hash = 0u;
                    motion_change = true;
                }
                continue;
            }

            RayTrophiSim::Fluid::SourceSplatRenderPlan source_plan(
                domains[d], system.runtime->flowSources(), static_cast<int>(d));
            const auto dconfig = source_plan.renderDomain(domains[d]);

            InstanceGroup* group = (group_id >= 0) ? im.getGroup(group_id) : nullptr;
            if (!group) {
                const std::string gname = "[FluidDomParticles] " + system.name +
                                          " D" + std::to_string(d);
                group_id = im.createGroup(gname, "", {});
                group = im.getGroup(group_id);
                if (!group) continue;
                group->transient = true;
                group->sources.clear();
                system.domain_particle_pool_capacities[d] = 0;
                g_fluid_source_state.erase(group_id);
            }
            group->transient = true;
            if (group->rendered_rt_excluded != !has_explicit_splats) {
                group->rendered_rt_excluded = !has_explicit_splats;
                structural_change = true;
            }
            group->raster_sphere_candidate = dconfig.fluid_particle_geometry_mode != 1 &&
                !dconfig.fluid_params.pore_exchange.wet_appearance_enabled;
            const bool procedural_splats =
                dconfig.fluid_particle_geometry_mode == 0 && has_explicit_splats;
            if (group->point_sphere_mode != procedural_splats) {
                group->point_sphere_mode = procedural_splats;
                structural_change = true;
            }
            const uint32_t requested_visual_children =
                domains[d].fluid_visual_samples.enabled ? 1u : source_plan.visualChildren(domains[d]);
            const uint32_t effective_visual_children =
                RayTrophiSim::Fluid::limitFluidRenderProxyChildren(
                    requested_visual_children,
                    RayTrophiSim::Fluid::fluidRenderProxyCarrierCapacity(
                        dconfig.fluid_max_particles,
                        dconfig.fluid_foam_params.enabled
                            ? dconfig.fluid_foam_params.max_foam
                            : 0u),
                    dconfig.fluid_particle_visual_budget);
            const uint8_t visual_children = procedural_splats
                ? static_cast<uint8_t>(effective_visual_children)
                : 1u;
            if (group->point_sphere_visual_children != visual_children) {
                group->point_sphere_visual_children = visual_children;
                structural_change = true;
            }
            // A grain's size is its physical radius, never a visual jitter.
            const float visual_size_variation = procedural_splats &&
                    !dconfig.fluid_params.grain.enabled
                ? std::clamp(
                      dconfig.fluid_particle_visual_size_variation,
                      0.0f,
                      RayTrophiSim::Fluid::kFluidRenderProxyMaxSizeVariation)
                : 0.0f;
            if (group->point_sphere_visual_size_variation != visual_size_variation) {
                group->point_sphere_visual_size_variation = visual_size_variation;
                structural_change = true;
            }
            // ── Near grain meshes (GrainMeshNear.h, K0 measurement) ──────────
            // Evaluated before the skip test: a camera move re-splits a paused
            // simulation, and a config the split cannot honour says why.
            namespace GF = RayTrophiSim::Fluid;
            GF::GrainMeshNearStats near_stats;
            near_stats.enabled = dconfig.fluid_grain_mesh_near_enabled;
            near_stats.camera_distance = std::max(dconfig.fluid_grain_mesh_near_distance, 0.0f);
            bool near_active = false;
            Vec3 near_camera(0.0f, 0.0f, 0.0f);
            uint64_t near_camera_key = 0;
            if (near_stats.enabled) {
                const auto camera = getActiveCamera();
                if (!procedural_splats) {
                    near_stats.status = "needs geometry 'icosphere' (procedural spheres)";
                } else if (dconfig.fluid_grain_mesh_near_shape == "scene" &&
                           dconfig.fluid_grain_mesh_near_sources.empty()) {
                    near_stats.status = "no exemplar sources";
                } else if (!camera) {
                    near_stats.status = "no active camera";
                } else {
                    near_active = true;
                    near_camera = camera->lookfrom;
                    near_camera_key = GF::sourceSplatCameraKey(near_camera);
                }
            }
            if (!near_active) {
                const auto near_it = g_grain_mesh_groups.find(group_id);
                if (near_it != g_grain_mesh_groups.end()) near_it->second.release = true;
            }

            // Source rebuild on geometry/material-routing changes. Scene mesh
            // faces keep their authored materials unless an explicit binding
            // requests a uniform override.
            uint64_t sig = source_plan.signature(world.objects);
            sig = hashCombine(sig, RayTrophiSim::Fluid::domainVisualSettingsKey(
                domains[d].fluid_visual_samples));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_color.x));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_color.y));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_color.z));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_radius_factor));
            sig = hashCombine(sig, GF::matterGrainSizeSignature(dconfig.fluid_params.grain));
            sig = hashCombine(sig, dconfig.fluid_params.grain.enabled ? 1ull : 0ull);
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_size_multiplier));
            sig = hashCombine(sig, static_cast<uint64_t>(
                std::max(dconfig.fluid_particle_visual_children, 1)));
            sig = hashCombine(
                sig, quantize(dconfig.fluid_particle_visual_size_variation));
            sig = hashCombine(
                sig,
                dconfig.fluid_granular_physical_carriers ? 1ull : 0ull);
            sig = hashCombine(sig, dconfig.fluid_particle_visual_budget);
            sig = hashCombine(sig, dconfig.fluid_params.granular_enabled ? 1ull : 0ull);
            sig = hashCombine(sig, static_cast<uint64_t>(
                std::max(dconfig.fluid_params.particles_per_cell, 1)));
            sig = hashCombine(sig, quantize(dconfig.fluid_particle_emission));
            sig = hashCombine(sig, dconfig.fluid_particle_emissive ? 1ull : 0ull);
            sig = hashCombine(sig, static_cast<uint64_t>(dconfig.fluid_render_mode));
            sig = hashCombine(sig, static_cast<uint64_t>(dconfig.fluid_max_particles));
            sig = hashCombine(sig, needs_raster_sdf_proxy ? 1ull : 0ull);
            sig = hashCombine(sig, static_cast<uint64_t>(
                std::max(0, std::min(dconfig.fluid_particle_subdivisions, 3))));
            sig = hashCombine(sig, static_cast<uint64_t>(dconfig.fluid_particle_geometry_mode));
            const bool scene_source_live =
                dconfig.fluid_particle_geometry_mode == 1 &&
                !dconfig.fluid_particle_geometry_source.empty() &&
                hasLiveSimulationObject(dconfig.fluid_particle_geometry_source);
            sig = hashCombine(sig, scene_source_live ? 1ull : 0ull);
            sig = hashCombine(sig, static_cast<uint64_t>(
                static_cast<uint32_t>(dconfig.fluid_particle_material_id)));
            for (unsigned char c : dconfig.fluid_particle_geometry_source)
                sig = hashCombine(sig, static_cast<uint64_t>(c));
            // User-assigned materials must invalidate the cached BLAS — different
            // ID = different shader binding on the backends.
            for (const auto& b : dconfig.fluid_substance_materials) {
                sig = hashCombine(sig, RayTrophiSim::Fluid::substanceTag(b.substance));
                sig = hashCombine(sig, static_cast<uint64_t>(static_cast<uint32_t>(b.material_id)));
                sig = hashCombine(sig, static_cast<uint64_t>(b.representation));
            }
            for (const auto route : dconfig.fluid_label_routes) {
                sig = hashCombine(sig, static_cast<uint64_t>(route));
            }
            sig = hashCombine(sig, near_active ? 1ull : 0ull);
            sig = hashCombine(sig, dconfig.fluid_grain_mesh_near_source_materials ? 1ull : 0ull);
            for (unsigned char c : dconfig.fluid_grain_mesh_near_shape) {
                sig = hashCombine(sig, c);
            }
            for (const auto& name : dconfig.fluid_grain_mesh_near_sources) {
                for (unsigned char c : name) sig = hashCombine(sig, static_cast<uint64_t>(c));
                sig = hashCombine(sig, 0x5bd1e995ull);  // separator
            }



            const auto wet_palette = RayTrophiSim::Fluid::prepareMatterWetPalette(
                dconfig, render_particles);
            for (const auto& entry : wet_palette.entries) {
                for (const auto material : entry.materials) {
                    sig = hashCombine(sig, static_cast<uint64_t>(material));
                }
            }
            sig = hashCombine(sig,
                RayTrophiSim::Fluid::matterPoreSettingsHash(dconfig.fluid_params.pore_exchange));
            FluidSourceState& st = g_fluid_source_state[group_id];
            if (!group->sources.empty() && st.signature == sig &&
                st.state_version == state.version && st.content_hash != 0u &&
                st.near_camera_key == near_camera_key) {
                if (near_active) {
                    const auto near_it = g_grain_mesh_groups.find(group_id);
                    if (near_it != g_grain_mesh_groups.end()) near_it->second.touched = true;
                }
                continue;
            }
            st.near_camera_key = near_camera_key;
            if (group->sources.empty() || st.signature != sig ||
                st.source_material_keys.size() != group->sources.size()) {
                group->sources.clear();
                const std::string geo_node = "[FluidDomGeo] " + system.name +
                                              " D" + std::to_string(d);
                // A valid explicit material wins. Built-in geometry otherwise
                // reuses an existing scene material; selected scene geometry
                // uses its original per-face material IDs.
                const bool scene_geometry = scene_source_live;
                const uint16_t fallback_mat =
                    RayTrophiSim::Fluid::resolveExistingSplatMaterial(
                        dconfig.fluid_particle_material_id);
                const int base_material_key =
                    RayTrophiSim::Fluid::isExistingSplatMaterial(
                        dconfig.fluid_particle_material_id)
                        ? dconfig.fluid_particle_material_id
                        : (scene_geometry ? -1 : static_cast<int>(fallback_mat));
                // One geometry source per distinct splat material. Instances
                // select a source; material IDs are never interpolated.
                std::vector<int> splat_material_keys{base_material_key};
                for (const auto& b : dconfig.fluid_substance_materials) {
                    // Material binding is independent from representation. An
                    // Inherit binding in a Splat-default domain still owns its
                    // material; filtering it here made those particles silently
                    // fall through to the domain material.
                    const int material_key =
                        RayTrophiSim::Fluid::isExistingSplatMaterial(b.material_id)
                            ? b.material_id : base_material_key;
                    if (std::find(splat_material_keys.begin(), splat_material_keys.end(), material_key) ==
                        splat_material_keys.end()) {
                        splat_material_keys.push_back(material_key);
                    }
                }
                wet_palette.appendMaterialKeys(splat_material_keys);
                const int subdiv = std::max(0, std::min(dconfig.fluid_particle_subdivisions, 3));
                for (std::size_t si = 0; si < splat_material_keys.size(); ++si) {
                    const int material_key = splat_material_keys[si];
                    const uint16_t geometry_mat = material_key >= 0
                        ? static_cast<uint16_t>(material_key) : fallback_mat;
                    ScatterSource src;
                    src.name = geo_node + (material_key >= 0
                        ? " M" + std::to_string(material_key) : " SceneMaterials");
                    src.weight = 1.0f;
                    if (scene_geometry) {
                        if (domains[d].fluid_visual_samples.enabled) {
                            std::vector<RayTrophiSim::Fluid::GrainExemplar> exemplars;
                            std::string error;
                            if (RayTrophiSim::Fluid::buildGrainExemplars(world.objects,
                                    {dconfig.fluid_particle_geometry_source}, exemplars, error)) {
                                for (const auto& exemplar : exemplars) {
                                    src.flat_meshes.push_back(
                                        RayTrophiSim::Fluid::buildDomainVisualFlatMesh(exemplar,
                                            material_key, src.name, exemplar.volume_scale));
                                }
                            }
                        } else {
                            gatherSceneMeshSource(world.objects,
                                              dconfig.fluid_particle_geometry_source,
                                              src.triangles);
                        }
                    }
                    // Missing/renamed scene nodes fall back visibly and safely;
                    // the UI keeps the unresolved name so the user can repair it.
                    const bool using_scene_mesh = !src.triangles.empty() || !src.flat_meshes.empty();
                    if (!using_scene_mesh)
                        buildIcosphere(subdiv, geometry_mat, src.name, src.triangles);
                    if (!using_scene_mesh || material_key >= 0) {
                        for (auto& tri : src.triangles)
                            if (tri) tri->setMaterialID(geometry_mat);
                    }
                    src.computeCenter();
                    group->sources.push_back(std::move(src));
                }
                st.signature = sig;
                st.source_capacities.assign(group->sources.size(), 0);
                st.source_material_keys = std::move(splat_material_keys);
                structural_change = true;
            }

            // Near mesh sources: one per (material slot, exemplar), so the
            // material axis (substance binding, wet band) stays the primary
            // pool's and the shape axis is the exemplar's.
            GrainMeshGroupState* near_state = nullptr;
            InstanceGroup* near_group = nullptr;
            if (near_active) {
                GrainMeshGroupState& ms = g_grain_mesh_groups[group_id];
                ms.release = false;
                const bool ready = ms.signature == sig && ms.group_id >= 0 &&
                    im.getGroup(ms.group_id) != nullptr &&
                    ms.material_keys == st.source_material_keys;
                if (!ready) {
                    std::string error;
                    if (!source_plan.build(world.objects, ms.exemplars, error)) {
                        near_active = false;
                        near_stats.status = error;
                        ms.release = true;
                    } else {
                        if (ms.group_id < 0 || !im.getGroup(ms.group_id)) {
                            ms.group_id = im.createGroup("[FluidDomGrainNear] " + system.name +
                                                         " D" + std::to_string(d), "", {});
                            // createGroup may reallocate the group vector.
                            group = im.getGroup(group_id);
                        }
                        InstanceGroup* mg = im.getGroup(ms.group_id);
                        if (!mg || !group) {
                            near_active = false;
                            near_stats.status = "instance group allocation failed";
                            ms.release = true;
                        } else {
                            mg->transient = true;
                            mg->point_sphere_mode = false;
                            mg->rendered_rt_excluded = false;
                            mg->raster_sphere_candidate = false;
                            mg->sources.clear();
                            const uint16_t fallback_mat = GF::resolveExistingSplatMaterial(
                                dconfig.fluid_particle_material_id);
                            for (const int key : st.source_material_keys) {
                                const uint16_t mat = key >= 0 ? static_cast<uint16_t>(key)
                                                              : fallback_mat;
                                for (const auto& exemplar : ms.exemplars) {
                                    ScatterSource src;
                                    src.name = "[FluidDomGrainNearGeo] " + system.name + " D" +
                                               std::to_string(d) + " M" + std::to_string(key) +
                                               " " + exemplar.node;
                                    src.weight = 1.0f;
                                    if (domains[d].fluid_visual_samples.enabled) {
                                        src.flat_meshes.push_back(GF::buildDomainVisualFlatMesh(
                                            exemplar, exemplar.source_materials ? -1 : mat,
                                            src.name));
                                    } else {
                                        GF::buildGrainExemplarTriangles(exemplar,
                                            exemplar.source_materials ? -1 : mat,
                                            src.name, src.triangles);
                                    }
                                    src.computeCenter();
                                    mg->sources.push_back(std::move(src));
                                }
                            }
                            ms.source_capacities.assign(mg->sources.size(), 0);
                            ms.material_keys = st.source_material_keys;
                            ms.signature = sig;
                            ms.content_hash = 0;
                            structural_change = true;
                        }
                    }
                }
                if (!group) continue;
                if (near_active) {
                    near_state = &ms;
                    near_group = im.getGroup(ms.group_id);
                }
            }
            const std::size_t near_exemplars = near_state ? near_state->exemplars.size() : 0;
            std::vector<std::vector<std::size_t>> near_particles(
                near_group ? near_group->sources.size() : 0);
            if (near_state) source_plan.index(near_state->exemplars);
            auto particleId = [&](std::size_t pi) {
                return pi < render_particles.particle_id.size()
                    ? render_particles.particle_id[pi] : static_cast<uint64_t>(pi);
            };

            // ── Pool grows monotonically, in chunks, and is budget-capped. ───
            // Explicit splats track the UI Max Particles value. SurfaceSDF's
            // raster compatibility proxy has its own smaller ceiling below.
            // draw_count is what we render; cap is the chunk-rounded pool size.
            // Chunked growth prevents a structural BLAS rebuild on every spawn.
            std::vector<std::vector<std::size_t>> source_particles(group->sources.size());
            const std::size_t budget = fluidRenderBudget(domains[d].fluid_visual_samples.enabled
                ? std::max<uint64_t>(dconfig.fluid_max_particles,
                    domains[d].fluid_visual_samples.sampling.max_samples + state.particles.size())
                : dconfig.fluid_max_particles);
            std::size_t accepted_particles = 0;
            std::size_t accepted_proxy_particles = 0;
            const std::size_t proxy_budget =
                std::min(budget, kFluidSurfaceProxyInstanceCeiling);
            const std::size_t proxy_stride = proxy_budget > 0
                ? std::max<std::size_t>(
                      1, (render_particles.size() + proxy_budget - 1) / proxy_budget)
                : 1;
            for (std::size_t pi = 0; pi < render_particles.size(); ++pi) {
                const uint32_t tag = pi < render_particles.substance_tag.size()
                    ? render_particles.substance_tag[pi] : RayTrophiSim::Fluid::kSubstanceUntagged;
                // Per PARCEL: the substance's view, then its state label's route
                // (a spray parcel of a surface substance is a splat here).
                const RayTrophiSim::Fluid::FluidView view =
                    view_plan.viewForParticle(render_particles, pi);
                const bool splat = view == RayTrophiSim::Fluid::FluidView::Splat;
                std::size_t source_index = 0;
                for (const auto& b : dconfig.fluid_substance_materials) {
                    // ★ Untagged liquid IS the Default Substance (same rule as
                    // collectSubstanceMaterialEntries): without this a spray or
                    // foam parcel of a seeded domain splats in the DOMAIN material
                    // while its surface already shows the binding (2026-11-04).
                    const bool owns =
                        RayTrophiSim::Fluid::substanceTag(b.substance) == tag ||
                        (tag == RayTrophiSim::Fluid::kSubstanceUntagged &&
                         !b.substance.empty() &&
                         b.substance == dconfig.fluid_params.default_substance);
                    if (!owns) continue;
                    const int base_key = st.source_material_keys.empty()
                        ? 0 : st.source_material_keys.front();
                    const int requested_key =
                        RayTrophiSim::Fluid::isExistingSplatMaterial(b.material_id)
                            ? b.material_id : base_key;
                    for (std::size_t si = 0; si < st.source_material_keys.size(); ++si) {
                        if (st.source_material_keys[si] == requested_key) {
                            source_index = si;
                            break;
                        }
                    }
                    break;
                }
                source_index = wet_palette.sourceIndex(
                    render_particles, pi, st.source_material_keys, source_index);
                // Proxy spheres stand in for a SURFACE only; fog parcels are
                // drawn by the fog volume and get no sphere.
                const bool sampled_proxy =
                    view == RayTrophiSim::Fluid::FluidView::Surface && needs_raster_sdf_proxy &&
                    accepted_proxy_particles < proxy_budget &&
                    pi % proxy_stride == 0;
                if (near_group && splat && accepted_particles < budget) {
                    uint32_t exemplar = 0;
                    bool over_budget = false;
                    if (source_plan.select(render_particles, pi, near_camera,
                            exemplar, over_budget)) {
                        near_particles[source_index * near_exemplars + exemplar].push_back(pi);
                        ++near_stats.near_instances;
                        ++accepted_particles;
                        continue;
                    }
                    if (over_budget) ++near_stats.near_over_budget;
                }
                if ((splat || sampled_proxy) && accepted_particles < budget) {
                    source_particles[source_index].push_back(pi);
                    ++accepted_particles;
                    if (sampled_proxy) ++accepted_proxy_particles;
                }
            }

            if (st.source_capacities.size() != source_particles.size()) {
                st.source_capacities.assign(source_particles.size(), 0);
                structural_change = true;
            }
            std::size_t total_capacity = 0;
            for (std::size_t si = 0; si < source_particles.size(); ++si) {
                const std::size_t live = source_particles[si].size();
                const std::size_t want = fluidPoolCapacityFor(live, budget);
                std::size_t& source_cap = st.source_capacities[si];
                // Capacity changes move the following material ranges, so they
                // are structural. Normal spawn/reseed inside a range is refit-only.
                if (want > source_cap) {
                    source_cap = want;
                    structural_change = true;
                } else if (source_cap > kFluidPoolChunk && want * 4 <= source_cap) {
                    source_cap = want;
                    structural_change = true;
                }
                total_capacity += source_cap;
            }
            system.domain_particle_pool_capacities[d] = total_capacity;

            std::vector<InstanceTransform>& inst = group->instances;
            if (inst.size() != total_capacity) {
                inst.resize(total_capacity);
                structural_change = true;
            }

            const float voxel = std::max(1e-4f, RayTrophiSim::Fluid::liquidGrid(state).voxel_size);
            const RayTrophiSim::Fluid::FluidParticleVisualRadius visual_radius =
                RayTrophiSim::Fluid::resolveFluidParticleVisualRadius(
                    dconfig.fluid_params.granular_enabled,
                    dconfig.fluid_particle_geometry_mode != 1,
                    dconfig.fluid_params.particles_per_cell,
                    dconfig.fluid_particle_radius_factor,
                    dconfig.fluid_particle_size_multiplier);
            const float radius = std::max(
                1e-4f,
                dconfig.fluid_params.grain.enabled ? dconfig.fluid_params.grain.radius_m :
                    voxel * visual_radius.effective_voxels);
            const float visual_spread_radius = procedural_splats && !dconfig.fluid_params.grain.enabled
                ? std::max(
                      0.0f,
                      voxel * RayTrophiSim::Fluid::kFluidRenderProxyFillSupportVoxels -
                          radius / std::cbrt(static_cast<float>(visual_children)))
                : 0.0f;
            if (group->point_sphere_visual_spread_radius != visual_spread_radius) {
                group->point_sphere_visual_spread_radius = visual_spread_radius;
                structural_change = true;
            }
            const float diam = radius * 2.0f;  // unit-diameter primitives -> scale = diameter
            // A grain domain may also hold liquid parcels (one transport owner
            // per carrier): only granular carriers are drawn as physical grains.
            const bool grain_domain = dconfig.fluid_params.grain.enabled;
            const float liquid_diam = 2.0f * std::max(1e-4f,
                voxel * visual_radius.effective_voxels);

            uint64_t content = hashCombine(1469598103934665603ull, state.version);
            if (domains[d].fluid_visual_samples.enabled) {
                group->point_sphere_ids.assign(inst.size(), 0);
            } else {
                group->point_sphere_ids.clear();
            }
            std::size_t drawn = 0;
            std::size_t slot_base = 0;
            for (std::size_t si = 0; si < source_particles.size(); ++si) {
                const std::size_t source_cap = st.source_capacities[si];
                for (std::size_t local = 0; local < source_cap; ++local) {
                    InstanceTransform& tr = inst[slot_base + local];
                    tr.rotation = Vec3(0.0f, 0.0f, 0.0f);
                    tr.source_index = static_cast<int>(si);
                    if (local >= source_particles[si].size()) {
                        tr.position = Vec3(0.0f, 0.0f, 0.0f);
                        tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                        continue;
                    }
                    const std::size_t pi = source_particles[si][local];
                    if (!group->point_sphere_ids.empty()) {
                        group->point_sphere_ids[slot_base + local] = particleId(pi);
                    }
                    const Vec3& p = render_particles.position[pi];
                    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                        tr.position = Vec3(0.0f, 0.0f, 0.0f);
                        tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                        continue;
                    }
                    tr.position = p;
                    tr.rotation = GF::domainSplatRotation(domains[d], state, pi, tr.rotation);
                    const float d = GF::domainSplatDiameter(domains[d], state, pi, GF::splatCarrierDiameter(
                        dconfig, render_particles, pi, liquid_diam));
                    // A Spray parcel is alone: one drop, not a cluster of visual
                    // children (FluidRenderProxy.h, fluidRenderProxyIsolated).
                    const bool lone = procedural_splats && visual_children > 1 &&
                        pi < render_particles.flags.size() &&
                        RayTrophiSim::Fluid::particleLabel(render_particles.flags[pi]) ==
                            RayTrophiSim::Fluid::ParticleLabel::Spray;
                    tr.scale = Vec3(d, lone ? -d : d, d);
                    ++drawn;
                }
                slot_base += source_cap;
            }
            content = hashCombine(content, quantize(diam));
            content = hashCombine(content, grain_domain ? quantize(liquid_diam) : 0ull);
            content = hashCombine(content, static_cast<uint64_t>(drawn));
            content = hashCombine(content, near_camera_key);

            if (near_group && near_state) {
                GrainMeshGroupState& ms = *near_state;
                if (ms.source_capacities.size() != near_particles.size()) {
                    ms.source_capacities.assign(near_particles.size(), 0);
                    structural_change = true;
                }
                std::size_t near_capacity = 0;
                for (std::size_t s = 0; s < near_particles.size(); ++s) {
                    const std::size_t live = near_particles[s].size();
                    std::size_t& cap = ms.source_capacities[s];
                    if (live > cap || live * 2 < cap) {
                        const std::size_t want = nearPoolCapacityFor(live);
                        if (want != cap) {
                            cap = want;
                            structural_change = true;
                        }
                    }
                    near_capacity += cap;
                }
                auto& near_inst = near_group->instances;
                if (near_inst.size() != near_capacity) {
                    near_inst.resize(near_capacity);
                    structural_change = true;
                }
                std::size_t near_base = 0;
                if (domains[d].fluid_visual_samples.enabled) {
                    near_group->point_sphere_ids.assign(near_inst.size(), 0);
                } else {
                    near_group->point_sphere_ids.clear();
                }
                for (std::size_t s = 0; s < near_particles.size(); ++s) {
                    const auto& exemplar = ms.exemplars[s % near_exemplars];
                    for (std::size_t local = 0; local < ms.source_capacities[s]; ++local) {
                        InstanceTransform& tr = near_inst[near_base + local];
                        tr.source_index = static_cast<int>(s);
                        if (local >= near_particles[s].size()) {
                            tr.position = Vec3(0.0f, 0.0f, 0.0f);
                            tr.rotation = Vec3(0.0f, 0.0f, 0.0f);
                            tr.scale = Vec3(0.0f, 0.0f, 0.0f);
                            continue;
                        }
                        const std::size_t pi = near_particles[s][local];
                        if (!near_group->point_sphere_ids.empty()) {
                            near_group->point_sphere_ids[near_base + local] = particleId(pi);
                        }
                        tr.position = render_particles.position[pi];
                        tr.rotation = GF::domainSplatRotation(domains[d], state, pi,
                            exemplar.random_orientation
                                ? GF::grainEulerDegreesFor(particleId(pi)) : Vec3(0.0f));
                        const float d_mesh = GF::domainSplatDiameter(domains[d], state, pi, GF::splatCarrierDiameter(
                            dconfig, render_particles, pi, liquid_diam)) * exemplar.volume_scale;
                        tr.scale = Vec3(d_mesh, d_mesh, d_mesh);
                    }
                    near_base += ms.source_capacities[s];
                }
                const uint64_t near_content = hashCombine(
                    hashCombine(content, 0x4e454152ull), static_cast<uint64_t>(near_capacity));
                if (structural_change || near_content != ms.content_hash) {
                    ms.content_hash = near_content;
                    near_group->gpu_dirty = true;
                    if (!structural_change) motion_change = true;
                }
                ms.touched = true;

                near_stats.status = "active";
                near_stats.exemplars = static_cast<uint32_t>(near_exemplars);
                double volume_scale_sum = 0.0;
                for (const auto& exemplar : ms.exemplars) {
                    near_stats.exemplar_triangles += exemplar.faces.size();
                    volume_scale_sum += exemplar.volume_scale;
                    if (!exemplar.volume_measured) ++near_stats.exemplars_without_volume;
                }
                near_stats.mean_volume_scale = near_exemplars > 0
                    ? static_cast<float>(volume_scale_sum / static_cast<double>(near_exemplars))
                    : 1.0f;
                near_stats.mesh_sources = static_cast<uint32_t>(near_particles.size());
                near_stats.mesh_slots = near_capacity;
            }
            near_stats.sphere_instances = drawn;
            GF::publishGrainMeshNearStats(dconfig.name, near_stats);

            if (structural_change) {
                st.content_hash = content;
                st.state_version = state.version;
                group->gpu_dirty = true;
            } else if (content != st.content_hash) {
                st.content_hash = content;
                st.state_version = state.version;
                group->gpu_dirty = true;
                motion_change = true;
            } else {
                st.state_version = state.version;
            }
        }
    }

    // Near grain pools whose domain was not split this pass: deleted when the
    // split was disabled or failed (or the primary pool is gone); only hidden
    // when the domain is merely hidden or empty - deleting and recreating a
    // pool under a rewind is the structural churn the primary pool avoids.
    for (auto it = g_grain_mesh_groups.begin(); it != g_grain_mesh_groups.end();) {
        GrainMeshGroupState& ms = it->second;
        InstanceGroup* mg = ms.group_id >= 0 ? im.getGroup(ms.group_id) : nullptr;
        if (im.getGroup(it->first) == nullptr || ms.release) {
            if (mg) {
                im.deleteGroup(ms.group_id);
                structural_change = true;
            }
            it = g_grain_mesh_groups.erase(it);
            continue;
        }
        if (!ms.touched && mg) {
            bool had_visible = false;
            for (auto& tr : mg->instances) {
                had_visible |= tr.scale.x != 0.0f || tr.scale.y != 0.0f || tr.scale.z != 0.0f;
                tr.scale = Vec3(0.0f);
            }
            if (had_visible) {
                mg->gpu_dirty = true;
                ms.content_hash = 0;
                motion_change = true;
            }
        }
        ms.touched = false;
        ++it;
    }

    if (structural_change) {
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_viewport_raster_rebuild_pending = true;   // refresh Solid/Matcap viewport too
        // Bump the scene geometry generation token — the backends gate their
        // accel/raster rebuild on it (VulkanBackend::buildRasterGeometryImpl
        // skips when m_rasterBuiltGeometryGeneration matches). Without the bump
        // a structural fluid change set the dirty bools but the backend saw an
        // unchanged generation and rebuilt the TLAS against a stale BLAS/source
        // set — a prime suspect for the cross-backend corruption.
        g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    } else if (motion_change) {
        g_gpu_refit_pending = true;
    }
    if (structural_change || motion_change) g_particle_cpu_geometry_dirty = true;
}

