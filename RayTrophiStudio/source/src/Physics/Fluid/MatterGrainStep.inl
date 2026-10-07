// Included in ParticleSimulation.cpp's private namespace, after MatterGpuStep.inl.
//
// One frame of a grain-enabled Matter domain. Every carrier has exactly one
// transport owner for the frame: granular carriers -> the DEM grain step,
// liquid carriers -> the mixed Vulkan liquid lane (P2G/MGPCG/G2P). The two
// owners never step the same carrier and never see each other's state except
// through the coupling below, which is staggered per frame:
//   1. liquid subset: forces, pressure projection, advection (frame dt);
//   2. bin the liquid end state + grain volumes on the domain grid;
//   3. grain subset: DEM substeps with implicit drag against a private liquid
//      lump per grain + Archimedes buoyancy;
//   4. the equal and opposite impulse goes back to the liquid parcels.
// Canonical particles change only after every stage succeeded.
// Analytic colliders as triangles for the grain BVH. 24 x 12 facets keep the
// chord sag under 1% of the radius (a grain of radius r sees a facet error
// below r for colliders up to ~100 r).
void appendGrainSphereCollider(const Vec3& centre, float radius,
                               std::vector<SurfaceMeshTriangle>& out) {
    constexpr int slices = 24, stacks = 12;
    const auto point = [&](int stack, int slice) {
        const float theta = 3.14159265f * stack / stacks;
        const float phi = 6.28318531f * slice / slices;
        return centre + Vec3(std::sin(theta) * std::cos(phi), std::cos(theta),
                             std::sin(theta) * std::sin(phi)) * radius;
    };
    for (int a = 0; a < stacks; ++a) {
        for (int b = 0; b < slices; ++b) {
            SurfaceMeshTriangle t;
            t.p0 = point(a, b); t.p1 = point(a + 1, b); t.p2 = point(a + 1, b + 1);
            if (a + 1 < stacks) out.push_back(t);
            t.p1 = point(a + 1, b + 1); t.p2 = point(a, b + 1);
            if (a > 0) out.push_back(t);
        }
    }
}

void appendGrainCapsuleCollider(const Vec3& start, const Vec3& end, float radius,
                                std::vector<SurfaceMeshTriangle>& out) {
    // Two hemispheres (as spheres; the inner halves are buried) + the tube.
    appendGrainSphereCollider(start, radius, out);
    appendGrainSphereCollider(end, radius, out);
    const Vec3 axis = end - start;
    const float length = axis.length();
    if (!(length > 1e-6f)) return;
    const Vec3 w = axis / length;
    const Vec3 helper = std::abs(w.y) < .9f ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    const Vec3 u = Vec3::cross(helper, w).normalize();
    const Vec3 v = Vec3::cross(w, u);
    constexpr int slices = 24;
    for (int b = 0; b < slices; ++b) {
        const float a0 = 6.28318531f * b / slices, a1 = 6.28318531f * (b + 1) / slices;
        const Vec3 r0 = (u * std::cos(a0) + v * std::sin(a0)) * radius;
        const Vec3 r1 = (u * std::cos(a1) + v * std::sin(a1)) * radius;
        SurfaceMeshTriangle t;
        t.p0 = start + r0; t.p1 = end + r0; t.p2 = end + r1;
        out.push_back(t);
        t.p1 = end + r1; t.p2 = start + r1;
        out.push_back(t);
    }
}

void appendGrainBoxCollider(const Vec3& low, const Vec3& high,
                            std::vector<SurfaceMeshTriangle>& out) {
    const auto corner = [&](int i) {
        return Vec3(i & 1 ? high.x : low.x, i & 2 ? high.y : low.y, i & 4 ? high.z : low.z);
    };
    static const int faces[6][4] = {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1},
                                    {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
    for (const auto& f : faces) {
        SurfaceMeshTriangle t;
        t.p0 = corner(f[0]); t.p1 = corner(f[1]); t.p2 = corner(f[2]);
        out.push_back(t);
        t.p1 = corner(f[2]); t.p2 = corner(f[3]);
        out.push_back(t);
    }
}

bool runMatterGrainStep(SimulationGridDomainState& state,
    const Fluid::APICSolverParams& params, bool legacy_granular, float dt, float time_seconds,
    SimulationComputeContext* compute, SimulationGridDomainComputeBuffers& buffers,
    const std::function<bool(SimulationGridDomainComputeBuffers&)>& ensure,
    MatterExchangeLedger& ledger,
    const std::vector<ParticleColliderDesc>& colliders,
    const std::function<bool(const ParticleColliderDesc&,
        std::vector<SurfaceMeshTriangle>&, uint64_t&)>& resolve_mesh,
    bool unsupported_motion_or_fields, std::string& error) {
    if (!compute || params.boundary != Fluid::APICSolverParams::BoundaryMode::Closed ||
        params.pore_exchange.enabled || params.pore_exchange.wet_response_enabled ||
        params.thermal_liquid_enabled || unsupported_motion_or_fields) {
        error = "dry grain candidate requires Closed Vulkan, static colliders, gravity only; "
            "pore/wet/thermal coupling disabled";
        return false;
    }
    std::vector<SurfaceMeshTriangle> triangles;
    Vec3 low, high;
    state.grid.getWorldBounds(low, high);
    for (const auto& collider : colliders) {
        if (!collider.enabled || !collider.fluid_collision_enabled) {
            continue;
        }
        if (collider.source_mode == ParticleColliderSourceMode::PlaneY) {
            if (std::abs(collider.plane_y - low.y) < 1e-6f) {
                continue;
            }
            SurfaceMeshTriangle a, b;
            a.p0 = Vec3(low.x, collider.plane_y, low.z);
            a.p1 = Vec3(high.x, collider.plane_y, low.z);
            a.p2 = Vec3(high.x, collider.plane_y, high.z);
            b.p0 = a.p0;
            b.p1 = a.p2;
            b.p2 = Vec3(low.x, collider.plane_y, high.z);
            triangles.insert(triangles.end(), {a, b});
        } else if (collider.source_mode == ParticleColliderSourceMode::ObjectMeshSDF ||
                   collider.source_mode == ParticleColliderSourceMode::ObjectMeshBVH) {
            std::vector<SurfaceMeshTriangle> flat;
            uint64_t revision = 0;
            if (!resolve_mesh || !resolve_mesh(collider, flat, revision) || flat.empty()) {
                error = "grain flat collider geometry unavailable: " + collider.name;
                return false;
            }
            triangles.insert(triangles.end(), flat.begin(), flat.end());
        } else if (collider.source_mode == ParticleColliderSourceMode::Sphere) {
            appendGrainSphereCollider(collider.sphere_center, collider.sphere_radius, triangles);
        } else if (collider.source_mode == ParticleColliderSourceMode::Capsule) {
            appendGrainCapsuleCollider(collider.capsule_start, collider.capsule_end,
                collider.capsule_radius, triangles);
        } else if (collider.source_mode == ParticleColliderSourceMode::ObjectAABB ||
                   collider.source_mode == ParticleColliderSourceMode::ObjectOBB ||
                   collider.source_mode == ParticleColliderSourceMode::ObjectConvexDecomp) {
            // Grains collide with triangles: an object-bound collider uses the
            // object's own surface (a box object is its OBB; AABB keeps the
            // world box of that surface). Without an object, the authored box.
            std::vector<SurfaceMeshTriangle> surface;
            uint64_t revision = 0;
            const bool has_surface = !collider.source_name.empty() && resolve_mesh &&
                resolve_mesh(collider, surface, revision) && !surface.empty();
            if (collider.source_mode == ParticleColliderSourceMode::ObjectAABB || !has_surface) {
                Vec3 box_low = collider.bounds_min, box_high = collider.bounds_max;
                if (has_surface) {
                    box_low = box_high = surface.front().p0;
                    for (const auto& t : surface) {
                        for (const Vec3& v : {t.p0, t.p1, t.p2}) {
                            box_low = Vec3::min(box_low, v);
                            box_high = Vec3::max(box_high, v);
                        }
                    }
                }
                appendGrainBoxCollider(box_low, box_high, triangles);
            } else {
                triangles.insert(triangles.end(), surface.begin(), surface.end());
            }
        } else {
            error = "grain collider type not supported: " + collider.name;
            return false;
        }
        if (triangles.size() > 4096) {
            error = "grain candidate flat collider budget is 4096 faces; no face decimation";
            return false;
        }
    }

    const auto order_start = std::chrono::steady_clock::now();
    const auto ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    std::vector<std::size_t> liquid_order, grain_order;
    if (!Fluid::partitionMatterGrainOwners(state.particles, legacy_granular,
            params.grain.wet_grains, liquid_order, grain_order, error)) {
        return false;
    }
    Fluid::orderMatterGrainsByCell(state.particles, state.grid.origin,
        2.0f * params.grain.radius_m, grain_order);
    auto liquid = Fluid::selectMatterParticles(state.particles, liquid_order);
    auto grains = Fluid::selectMatterParticles(state.particles, grain_order);
    Fluid::APICSolverStats stats;
    Fluid::MatterGrainStepReport report;
    report.host_order_ms = ms_since(order_start);
    report.liquid_parcels = liquid.size();
    const auto budget = params.mixed_working_set_budget_bytes;
    const bool coupled = !liquid.empty() && !grains.empty() && params.grain.fluid_coupling;
    // B5: the liquid projection sees the grains' volume; the grains then take
    // the force the liquid's own acceleration implies, rho V (Du/Dt - g),
    // instead of hydrostatic buoyancy.
    const bool exclude = coupled && params.grain.volume_exclusion;
    // Each parcel's velocity before the liquid step (after last frame's grain
    // reaction): Du/Dt is measured per parcel, Lagrangian, by identity.
    const auto liquid_start_velocity = exclude ? liquid.velocity : std::vector<Vec3>{};
    const auto liquid_start_id = exclude ? liquid.particle_id : std::vector<uint64_t>{};

    // 1. Liquid owner. The mixed driver steps state.particles, so the liquid
    // subset is swapped in for the call and swapped back on every path.
    if (!liquid.empty()) {
        struct SwapBack {
            Fluid::FluidParticles& canonical;
            Fluid::FluidParticles& subset;
            ~SwapBack() { std::swap(canonical, subset); }
        };
        std::swap(state.particles, liquid);
        SwapBack swap_back{state.particles, liquid};
        if (!ensure(buffers)) {
            error = "grain domain: liquid lane GPU buffers could not be allocated";
            return false;
        }
        // The porous weights and grain velocities live in the grid only for
        // this liquid step; the collider weight cache sees its own values back.
        struct RestorePorosity {
            FluidSim::FluidGrid& grid;
            Fluid::MatterGrainPorosityBackup backup;
            SimulationGridDomainComputeBuffers& buffers;
            ~RestorePorosity() {
                Fluid::restoreMatterGrainPorosity(grid, backup);
                buffers.porous_solid_velocity = false;
            }
        } porosity{state.grid, {}, buffers};
        // Grains meet colliders as spheres; liquid parcels still need the
        // solid-cell recovery the plain liquid path runs before P2G.
        Fluid::recoverParticlesFromSolidCells(state.particles, state.grid);
        auto liquid_params = params;
        liquid_params.granular_enabled = false;
        if (exclude) {
            // Pore fraction never below .3: a packed bed is ~.36-.4, and a
            // face closed further would stall the projection's CG.
            Fluid::applyMatterGrainPorosity(state.grid, grains, params.grain.radius_m,
                params.variational_solids, .3f, porosity.backup, report);
            buffers.porous_solid_velocity = true;
            liquid_params.variational_solids = true;
        }
        if (!runGpuFluidParticleIntegrateForces(state, liquid_params, Vec3(0.0f, 0.0f, 0.0f), dt,
                time_seconds, nullptr, compute, buffers)) {
            error = "grain domain: liquid force integration failed";
            return false;
        }
        liquid_params.external_forces_preintegrated = true;
        if (!runMatterGpuStep(state, liquid_params, legacy_granular, dt, compute, buffers,
                ensure, ledger, error)) {
            error = "grain domain liquid lane: " + error;
            return false;
        }
        stats = state.fluid_stats;
    }

    // 2-4. Grain owner with the liquid coupling.
    if (!grains.empty()) {
        if (!buffers.matter_runtime) {
            buffers.matter_runtime = std::make_shared<Fluid::MatterGpuRuntime>();
        }
        const auto coupling_start = std::chrono::steady_clock::now();
        Fluid::MatterGrainCouplingFrame frame;
        if (coupled) {
            if (!Fluid::buildMatterGrainLiquidField(liquid, grains, params.grain.radius_m,
                    params.chemistry_preset, state.grid.origin, state.grid.nx, state.grid.ny,
                    state.grid.nz, state.grid.voxel_size, frame.field, error)) {
                return false;
            }
            Fluid::prepareMatterGrainCoupling(grains, params.grain, params.gravity, dt,
                frame, report);
            if (exclude && !Fluid::applyMatterGrainLiquidAccelerationForce(liquid,
                    liquid_start_velocity, liquid_start_id, grains, params.grain.radius_m,
                    params.gravity, dt, frame, report, error)) {
                return false;
            }
            report.volume_exclusion = exclude;
        }
        float coupling_ms = ms_since(coupling_start);
        std::vector<Fluid::MatterGrainCouplingOutput> drag;
        const std::size_t grain_budget = budget > stats.mixed_working_set_bytes
            ? budget - stats.mixed_working_set_bytes : budget;
        if (budget && stats.mixed_working_set_bytes >= budget) {
            error = "grain domain: liquid lane used the whole resource budget";
            return false;
        }
        if (!Fluid::stepMatterGrainGpu(grains, low, high, params.grain, dt, params.gravity,
                grain_budget, *compute, buffers.matter_runtime->grain, triangles,
                coupled ? &frame.inputs : nullptr, coupled ? &drag : nullptr, report, error)) {
            return false;
        }
        report.coupling_enabled = coupled;
        const auto reaction_start = std::chrono::steady_clock::now();
        if (coupled) {
            Fluid::applyMatterGrainLiquidReaction(liquid, frame, drag, report);
        }
        // B6: after the reaction, so the reaction used the masses it was
        // computed with. Without liquid (or coupling) only drying runs.
        Fluid::exchangeMatterGrainWater(liquid, grains, frame, params.grain, dt, report);
        report.host_coupling_ms = coupling_ms + ms_since(reaction_start);
    }

    const auto merge_start = std::chrono::steady_clock::now();
    if (!Fluid::mergeMatterGrainOwners(state.particles, liquid, grains, error)) {
        return false;
    }
    report.host_merge_ms = ms_since(merge_start);
    report.grains = grains.size();
    stats.particle_count = state.particles.size();
    stats.grain_report = report;
    stats.mixed_model_step = true;
    stats.mixed_step_held = false;
    stats.mixed_common_substeps = std::max(stats.mixed_common_substeps, report.substeps);
    stats.mixed_working_set_bytes += report.working_set_bytes;
    if (liquid.empty()) {
        stats.mixed_contact_pairs = report.contacts;
        stats.pressure_on_gpu = stats.g2p_on_gpu = stats.p2g_on_gpu = false;
    }
    stats.gpu_status = liquid.empty()
        ? "Dry grain Vulkan DEM candidate: fused hash/contact/history/spin"
        : grains.empty() ? "Grain domain: liquid lane only (Vulkan indexed transfer)"
        : std::string("Grain + liquid: DEM grains, Vulkan liquid lane, ") +
            (!report.coupling_enabled ? "coupling OFF (pass-through)"
             : report.volume_exclusion ? "porous projection + pressure force + implicit drag"
                                       : "implicit drag + hydrostatic buoyancy");
    state.fluid_stats = stats;
    return true;
}
