#include "Fluid/DemBondGpu.h"
#include "Fluid/MatterGrain.h"
#include "Fluid/FluidGpuDispatch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace RayTrophiSim::Fluid {
DemBondGpuFrame::DemBondGpuFrame(SimulationComputeContext& context, MatterGrainGpuRuntime& runtime)
    : compute(context), owned(runtime.bond_buffers) {}

DemBondGpuFrame::~DemBondGpuFrame() {
    // Device allocations belong to the runtime pool, not a speculative attempt.
    // Reuse them next frame/retry; avoid device-idle destruction every frame.
}

bool DemBondGpuFrame::prepare(const FluidParticles& p, const MatterGrainParams& params,
                             std::string& error) {
    active = !p.dem_bonds.nodes.empty();
    canonical_edge_count = p.dem_bonds.edges.size();
    if (!active) {
        if (canonical_edge_count != 0) {
            error = "DEM bonds: edge history has no endpoint nodes";
            return false;
        }
        for (auto& handle : owned) {
            if (handle.valid()) {
                compute.destroyBuffer(handle);
                handle = {};
            }
        }
        return true;
    }
    static_assert(sizeof(EdgeRow) == 96);
    static_assert(offsetof(EdgeRow, history) == 80);
    identities = p.particle_id;
    if (identities.size() != p.size() || p.size() > 10000000u) {
        error = "DEM bonds: invalid particle layout";
        return false;
    }
    std::unordered_map<uint64_t, uint32_t> index;
    // Zero is a transient nonmember marker, not a canonical orientation.
    poses.assign(p.size(), {0.0f, 0.0f, 0.0f, 0.0f});
    std::size_t bound_count = 0;
    for (uint32_t i = 0; i < p.size(); ++i) {
        if (!index.emplace(identities[i], i).second) {
            error = "DEM bonds: duplicate particle identity";
            return false;
        }
        const auto found = p.dem_bonds.nodes.find(identities[i]);
        if (found != p.dem_bonds.nodes.end()) {
            ++bound_count;
            for (int k = 0; k < 4; ++k) {
                poses[i][k] = static_cast<float>(found->second.orientation[k]);
            }
        }
    }
    if (bound_count != p.dem_bonds.nodes.size()) {
        error = "DEM bonds: unretired missing node";
        return false;
    }
    std::vector<std::vector<uint32_t>> links;
    std::vector<uint32_t> intact_degrees;
    for (std::size_t canonical = 0; canonical < canonical_edge_count; ++canonical) {
        const auto& edge = p.dem_bonds.edges[canonical];
        if (edge.state.failure != DemBond::Failure::None) {
            continue; // preserve history on the host, omit inert device rows
        }
        if (links.empty()) {
            links.resize(p.size());
            intact_degrees.resize(p.size(), 0);
        }
        const auto a = index.find(edge.state.first_id);
        const auto b = index.find(edge.state.second_id);
        if (a == index.end() || b == index.end()) {
            error = "DEM bonds: unretired missing endpoint";
            return false;
        }
        EdgeRow row;
        row.meta = {a->second, b->second, static_cast<uint32_t>(edge.state.failure), 0};
        for (int k = 0; k < 3; ++k) {
            row.rest[k] = static_cast<float>(edge.state.rest_separation[k]);
        }
        row.rest[3] = static_cast<float>(edge.state.rest_length_m);
        for (int k = 0; k < 4; ++k) {
            row.qa[k] = static_cast<float>(edge.state.first_rest_orientation[k]);
            row.qb[k] = static_cast<float>(edge.state.second_rest_orientation[k]);
        }
        const auto& m = edge.material;
        row.material = {float(m.young_pa), float(m.poisson), float(m.radius_m),
                        float(m.tensile_strength_pa)};
        row.history = {float(m.shear_strength_pa), float(edge.elastic_energy_j),
                       float(edge.state.fracture_dissipation_j), 0.0f};
        for (const auto* values : {&row.rest, &row.qa, &row.qb, &row.material, &row.history}) {
            for (float v : *values) {
                if (!std::isfinite(v)) {
                    error = "DEM bonds: state exceeds GPU numeric range";
                    return false;
                }
            }
        }
        const auto e = static_cast<uint32_t>(edges.size());
        links[a->second].push_back(e);
        links[b->second].push_back(e);
        ++intact_degrees[a->second];
        ++intact_degrees[b->second];
        edges.push_back(row);
        edge_indices.push_back(canonical);
    }
    constrains_motion = !edges.empty();
    uint32_t degree = 1;
    if (constrains_motion) {
        adjacency.resize(p.size() + 1);
        for (uint32_t i = 0; i < p.size(); ++i) {
            degree = std::max(degree, intact_degrees[i]);
            adjacency[i] = static_cast<uint32_t>(adjacency.size());
            adjacency.insert(adjacency.end(), links[i].begin(), links[i].end());
        }
        adjacency[p.size()] = static_cast<uint32_t>(adjacency.size());
    } else {
        adjacency.assign(1, 0); // unused by orientation-only shader, valid dummy binding
    }
    for (std::size_t canonical : edge_indices) {
        const auto& edge = p.dem_bonds.edges[canonical];
        const auto i = index.at(edge.state.first_id);
        const auto j = index.at(edge.state.second_id);
        const auto mass = [&](uint32_t k) {
            return double(p.rest_mass_kg[k]) * p.mass_fraction[k] +
                (params.wet_grains ? p.pore_water_mass_kg[k] : 0.0);
        };
        const double mi = mass(i), mj = mass(j);
        const double ri = matterGrainRadius(params, p, i);
        const double rj = matterGrainRadius(params, p, j);
        const double step = DemBond::stableStep(edge.material, edge.state.rest_length_m,
            mi, mj, 0.4 * mi * ri * ri, 0.4 * mj * rj * rj, degree, error);
        if (!(step > 0.0)) {
            return false;
        }
        maximum_dt = std::min(maximum_dt, step);
    }
    const std::array<std::size_t, 5> sizes{poses.size() * sizeof(poses[0]),
        std::max(edges.size(), std::size_t{1}) * sizeof(EdgeRow),
        adjacency.size() * sizeof(uint32_t),
        std::max(edges.size(), std::size_t{1}) * 32u, sizeof(status)};
    if (!constrains_motion) {
        // Drop large retired edge/load pools once, keeping valid tiny bindings
        // for the orientation-only kernel's unchanged descriptor layout.
        for (std::size_t i : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
            if (compute.getBufferSize(owned[i]) > sizes[i]) {
                compute.destroyBuffer(owned[i]);
                owned[i] = {};
            }
        }
    }
    upload_bytes = sizes[0] + edges.size() * sizeof(EdgeRow) + sizes[2] + sizes[4];
    download_bytes = sizes[0] + edges.size() * sizeof(EdgeRow) + sizes[4];
    for (std::size_t i = 0; i < owned.size(); ++i) {
        const auto capacity = compute.getBufferSize(owned[i]);
        working_bytes += capacity;
        if (capacity < sizes[i]) {
            working_bytes += sizes[i]; // replacement peak includes the old pool
        }
    }
    return true;
}

bool DemBondGpuFrame::dispatch(const char* kernel, MatterGrainGpuRuntime& runtime,
                              uint32_t substep, float dt, uint32_t threads,
                              std::string& error) {
    if (!active) {
        return true;
    }
    if (!uploaded) {
        const std::array<std::size_t, 5> sizes{poses.size() * sizeof(poses[0]),
            std::max(edges.size(), std::size_t{1}) * sizeof(EdgeRow),
            adjacency.size() * sizeof(uint32_t),
            std::max(edges.size(), std::size_t{1}) * 32u, sizeof(status)};
        for (std::size_t i = 0; i < owned.size(); ++i) {
            const auto limit = compute.caps().max_storage_buffer_bytes;
            if (limit != 0 && sizes[i] > limit) {
                error = "DEM bonds: buffer exceeds device storage range";
                return false;
            }
            if (compute.getBufferSize(owned[i]) < sizes[i]) {
                const auto handle = compute.createBuffer({"DemBondFrame", sizes[i],
                    ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                        ComputeBufferUsage::Download});
                if (!handle.valid()) {
                    error = "DEM bonds: frame buffer allocation failed";
                    return false;
                }
                if (owned[i].valid()) {
                    compute.destroyBuffer(owned[i]);
                }
                owned[i] = handle;
            }
        }
        compute.beginTransferBatch();
        bool ok = compute.uploadBuffer(owned[0], poses.data(), sizes[0]);
        if (!edges.empty()) {
            ok = compute.uploadBuffer(owned[1], edges.data(), edges.size() * sizeof(EdgeRow)) && ok;
        }
        ok = compute.uploadBuffer(owned[2], adjacency.data(), sizes[2]) && ok;
        ok = compute.uploadBuffer(owned[4], &status, sizeof(status)) && ok;
        ok = compute.endTransferBatch() && ok;
        if (!ok) {
            error = "DEM bonds: frame upload failed";
            return false;
        }
        uploaded = true;
    }
    struct Constants {
        uint32_t count, edges, substep, unused;
        float dt, padding[3];
    } constants{static_cast<uint32_t>(poses.size()), static_cast<uint32_t>(edges.size()),
        substep, 0, dt, {0.0f, 0.0f, 0.0f}};
    static_assert(sizeof(Constants) == 32);
    const ComputeBufferHandle bindings[]{runtime.positions, runtime.velocities,
        runtime.affines, runtime.scratch, runtime.mass,
        owned[0], owned[1], owned[2], owned[3], owned[4]};
    ComputeDispatch command;
    command.kernel = kernel;
    command.buffers = bindings;
    command.buffer_count = 10;
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    command.groups = FluidGpuDispatch::groups256(threads);
    if (!compute.dispatch(command)) {
        error = std::string("DEM bonds: dispatch failed: ") + kernel;
        return false;
    }
    return true;
}

bool DemBondGpuFrame::before(MatterGrainGpuRuntime& r, uint32_t step, float dt,
                            std::string& error) {
    if (!constrains_motion) {
        return true;
    }
    if (active && (!std::isfinite(dt) || dt <= 0.0f || dt > maximum_dt * 1.000001)) {
        error = "DEM bonds: common clock exceeds bond stability limit";
        return false;
    }
    return dispatch("sim_dem_bond_evaluate", r, step, dt,
               static_cast<uint32_t>(edges.size()), error) &&
        dispatch("sim_dem_bond_kick", r, step, dt,
                 static_cast<uint32_t>(poses.size()), error);
}

bool DemBondGpuFrame::after(MatterGrainGpuRuntime& r, uint32_t step, float dt,
                           std::string& error) {
    return dispatch("sim_dem_bond_orient", r, step, dt,
                    static_cast<uint32_t>(poses.size()), error);
}

bool DemBondGpuFrame::readback(std::string& error) {
    if (!active) {
        return true;
    }
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(owned[0], poses.data(), poses.size() * sizeof(poses[0]));
    if (!edges.empty()) {
        ok = compute.downloadBuffer(owned[1], edges.data(), edges.size() * sizeof(EdgeRow)) && ok;
    }
    ok = compute.downloadBuffer(owned[4], &status, sizeof(status)) && ok;
    ok = compute.endTransferBatch() && ok;
    if (!ok || status != 0) {
        if (!ok) {
            compute.synchronize(); // queued readback pointers must outlive failure cleanup
        }
        error = "DEM bonds: nonfinite response, orientation or GPU readback failure";
        return false;
    }
    completed = true;
    return true;
}

bool DemBondGpuFrame::publish(FluidParticles& p, std::string& error) {
    if (!active) {
        return true;
    }
    if (!completed || identities != p.particle_id ||
        canonical_edge_count != p.dem_bonds.edges.size()) {
        error = "DEM bonds: unvalidated frame or topology changed before publication";
        return false;
    }
    auto& next = p.dem_bonds;
    for (std::size_t i = 0; i < identities.size(); ++i) {
        const auto node = next.nodes.find(identities[i]);
        if (node == next.nodes.end()) {
            continue;
        }
        double norm = 0.0;
        for (float v : poses[i]) {
            norm += double(v) * v;
        }
        if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-4) {
            error = "DEM bonds: invalid material orientation";
            return false;
        }
    }
    for (std::size_t e = 0; e < edges.size(); ++e) {
        const auto& dest = next.edges[edge_indices[e]];
        const auto& row = edges[e];
        const double previous = dest.state.fracture_dissipation_j;
        if (row.meta[0] >= identities.size() || row.meta[1] >= identities.size() ||
            identities[row.meta[0]] != dest.state.first_id ||
            identities[row.meta[1]] != dest.state.second_id || row.meta[2] > 2 ||
            (dest.state.failure != DemBond::Failure::None &&
             row.meta[2] != static_cast<uint32_t>(dest.state.failure)) ||
            !std::isfinite(row.history[1]) || row.history[1] < 0.0f ||
            !std::isfinite(row.history[2]) || row.history[2] < previous * (1.0 - 1.0e-6)) {
            error = "DEM bonds: invalid fracture/energy state";
            return false;
        }
    }
    // Validation above performs no writes. The commit below cannot allocate or
    // fail, so publication is atomic without cloning the static graph again.
    for (std::size_t i = 0; i < identities.size(); ++i) {
        const auto node = next.nodes.find(identities[i]);
        if (node != next.nodes.end()) {
            for (int k = 0; k < 4; ++k) {
                node->second.orientation[k] = poses[i][k];
            }
        }
    }
    for (std::size_t e = 0; e < edges.size(); ++e) {
        auto& dest = next.edges[edge_indices[e]];
        const auto& row = edges[e];
        dest.state.failure = static_cast<DemBond::Failure>(row.meta[2]);
        dest.state.fracture_dissipation_j = std::max(
            dest.state.fracture_dissipation_j, double(row.history[2]));
        dest.elastic_energy_j = row.history[1];
    }
    return true;
}
} // namespace RayTrophiSim::Fluid
