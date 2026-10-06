// Included in ParticleSimulation.cpp's private namespace.
bool runMatterGrainStep(SimulationGridDomainState& state,
    const Fluid::APICSolverParams& params, float dt, SimulationComputeContext* compute,
    SimulationGridDomainComputeBuffers& buffers,
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
        } else {
            error = "grain candidate supports PlaneY and flat mesh colliders: " + collider.name;
            return false;
        }
        if (triangles.size() > 4096) {
            error = "grain candidate flat collider budget is 4096 faces; no face decimation";
            return false;
        }
    }
    return Fluid::stepMatterGrainGpu(state, params.grain, dt, params.gravity,
        params.mixed_working_set_budget_bytes, *compute, buffers, triangles, error);
}
