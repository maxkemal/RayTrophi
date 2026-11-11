#include "Fluid/DemBondNetwork.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/MatterGrain.h"
#include "Fluid/MatterSubstanceState.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace RayTrophiSim::Fluid::DemBond {
namespace {
constexpr std::size_t maximum_edges = 4000000;
constexpr std::size_t maximum_degree = 64;

uint64_t unsignedValue(const nlohmann::json& v) {
    if (v.is_number_unsigned()) {
        return v.get<uint64_t>();
    }
    if (v.is_number_integer() && v.get<int64_t>() >= 0) {
        return static_cast<uint64_t>(v.get<int64_t>());
    }
    throw std::runtime_error("expected unsigned integer");
}

double numberValue(const nlohmann::json& v) {
    if (!v.is_number()) {
        throw std::runtime_error("expected number");
    }
    return v.get<double>();
}

template <std::size_t N>
std::array<double, N> arrayValue(const nlohmann::json& v) {
    if (!v.is_array() || v.size() != N) {
        throw std::runtime_error("invalid vector/quaternion shape");
    }
    std::array<double, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = numberValue(v[i]);
    }
    return result;
}

Endpoint endpoint(const FluidParticles& p, std::size_t i) {
    const auto& v = p.position[i];
    return {p.particle_id[i], {v.x, v.y, v.z}};
}

bool validEdge(const Edge& edge, std::string& error) {
    if (!gpuMaterialRepresentable(edge.material)) {
        error = "DEM bond: material is outside GPU floating-point range";
        return false;
    }
    for (const auto* q : {&edge.state.first_rest_orientation,
                          &edge.state.second_rest_orientation}) {
        double norm = 0.0;
        for (double v : *q) {
            norm += v * v;
        }
        if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-8) {
            error = "DEM bond: invalid reference orientation";
            return false;
        }
    }
    if ((edge.state.failure != Failure::None && edge.elastic_energy_j != 0.0) ||
        (edge.state.failure == Failure::None && edge.state.fracture_dissipation_j != 0.0)) {
        error = "DEM bond: inconsistent fracture energy state";
        return false;
    }
    Endpoint a{edge.state.first_id, {0.0, 0.0, 0.0}, edge.state.first_rest_orientation};
    Endpoint b{edge.state.second_id, edge.state.rest_separation,
               edge.state.second_rest_orientation};
    State state = edge.state;
    Response response;
    return std::isfinite(edge.elastic_energy_j) && edge.elastic_energy_j >= 0.0 &&
        evaluate(a, b, edge.material, state, response, error);
}
} // namespace

bool gpuMaterialRepresentable(const Material& m) {
    for (double value : {m.young_pa, m.poisson, m.radius_m,
                          m.tensile_strength_pa, m.shear_strength_pa}) {
        if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
            return false;
        }
    }
    return static_cast<float>(m.young_pa) > 0.0f &&
        static_cast<float>(m.radius_m) > 0.0f &&
        static_cast<float>(m.poisson) > -1.0f && static_cast<float>(m.poisson) < 0.5f &&
        m.tensile_strength_pa >= 0.0 && m.shear_strength_pa >= 0.0;
}

bool bindBody(FluidParticles& p, const MatterGrainParams& params,
              const std::vector<uint64_t>& ids, const Material& material,
              double reach_ratio, double section_ratio, uint64_t byte_budget,
              uint64_t& body, std::string& error) {
    error.clear();
    if (ids.size() < 2 || ids.size() > maximum_edges ||
        p.dem_bonds.nodes.size() > maximum_edges - ids.size() ||
        p.particle_id.size() != p.size() ||
        !std::isfinite(reach_ratio) || reach_ratio < 1.0 || reach_ratio > 2.0 ||
        !std::isfinite(section_ratio) || section_ratio <= 0.0 || section_ratio > 1.0 ||
        p.dem_bonds.next_body == 0 ||
        p.dem_bonds.next_body == std::numeric_limits<uint64_t>::max()) {
        error = "DEM body: invalid identities, reach [1,2] or section ratio (0,1]";
        return false;
    }
    std::unordered_map<uint64_t, std::size_t> index;
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (!index.emplace(p.particle_id[i], i).second || p.particle_id[i] == 0) {
            error = "DEM body: duplicate or zero particle identity";
            return false;
        }
    }
    std::vector<std::size_t> selected;
    std::set<uint64_t> unique(ids.begin(), ids.end());
    if (unique.size() != ids.size()) {
        error = "DEM body: duplicate selected identity";
        return false;
    }
    double max_radius = 0.0;
    for (uint64_t id : unique) {
        const auto found = index.find(id);
        if (found == index.end() || p.dem_bonds.nodes.count(id) != 0) {
            error = "DEM body: missing identity or particle already belongs to a body";
            return false;
        }
        const auto i = found->second;
        if (i >= p.constitutive_model.size() ||
            p.constitutive_model[i] != static_cast<uint32_t>(MatterConstitutiveModel::Granular)) {
            error = "DEM body: selection must contain DEM-owned granular carriers";
            return false;
        }
        selected.push_back(i);
        max_radius = std::max(max_radius, double(matterGrainRadius(params, p, i)));
    }
    const double cell = 2.0 * reach_ratio * max_radius;
    if (!std::isfinite(cell) || !(cell > 0.0)) {
        error = "DEM body: invalid grain radius";
        return false;
    }
    using Key = std::tuple<int64_t, int64_t, int64_t>;
    std::map<Key, std::vector<std::size_t>> bins;
    const auto keyOf = [&](std::size_t i, Key& key) {
        std::array<int64_t, 3> k{};
        for (int axis = 0; axis < 3; ++axis) {
            const double v = std::floor(double(p.position[i][axis]) / cell);
            if (!std::isfinite(v) || std::abs(v) >= 9.0e15) {
                return false;
            }
            k[axis] = static_cast<int64_t>(v);
        }
        key = {k[0], k[1], k[2]};
        return true;
    };
    Network next = p.dem_bonds;
    const uint64_t new_body = next.next_body++;
    std::unordered_map<uint64_t, std::size_t> degrees;
    const auto first_edge = next.edges.size();
    for (std::size_t i : selected) {
        Key key;
        if (!keyOf(i, key)) {
            error = "DEM body: invalid position/hash extent";
            return false;
        }
        const auto [x, y, z] = key;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const auto found = bins.find({x + dx, y + dy, z + dz});
                    if (found == bins.end()) {
                        continue;
                    }
                    for (std::size_t j : found->second) {
                        const double ri = matterGrainRadius(params, p, i);
                        const double rj = matterGrainRadius(params, p, j);
                        if ((p.position[i] - p.position[j]).length() > reach_ratio * (ri + rj)) {
                            continue;
                        }
                        Edge edge;
                        edge.material = material;
                        edge.material.radius_m = section_ratio * std::min(ri, rj);
                        if (!create(endpoint(p, j), endpoint(p, i), edge.state, error) ||
                            !validEdge(edge, error)) {
                            return false;
                        }
                        if (++degrees[p.particle_id[i]] > maximum_degree ||
                            ++degrees[p.particle_id[j]] > maximum_degree ||
                            next.edges.size() >= maximum_edges) {
                            error = "DEM body: graph capacity exceeded; no bonds were published";
                            return false;
                        }
                        next.edges.push_back(edge);
                        const uint64_t bytes = next.edges.size() * sizeof(Edge) +
                            (next.nodes.size() + selected.size()) * (sizeof(Node) + 64u);
                        if (byte_budget != 0 && bytes > byte_budget) {
                            error = "DEM body: canonical graph exceeds working-set budget";
                            return false;
                        }
                    }
                }
            }
        }
        bins[key].push_back(i);
    }
    if (next.edges.size() == first_edge) {
        error = "DEM body: no neighbors in bonding reach";
        return false;
    }
    for (std::size_t i : selected) {
        if (degrees[p.particle_id[i]] == 0) {
            error = "DEM body: isolated selected particle; increase reach or fix sampling";
            return false;
        }
        next.nodes.emplace(p.particle_id[i], Node{new_body});
    }
    p.dem_bonds = std::move(next);
    body = new_body;
    return true;
}

void retain(Network& network, const std::vector<uint64_t>& live_ids) {
    if (network.nodes.empty()) {
        return;
    }
    const std::unordered_set<uint64_t> live(live_ids.begin(), live_ids.end());
    bool missing = false;
    for (const auto& [id, node] : network.nodes) {
        if (live.count(id) == 0) {
            missing = true;
            break;
        }
    }
    if (!missing) {
        return; // unchanged identity population: no edge scan or topology allocation
    }
    auto& edges = network.edges;
    edges.erase(std::remove_if(edges.begin(), edges.end(), [&](const Edge& edge) {
        const bool removed = live.count(edge.state.first_id) == 0 ||
            live.count(edge.state.second_id) == 0;
        if (removed) {
            network.retired_energy_j += edge.elastic_energy_j +
                edge.state.fracture_dissipation_j;
        }
        return removed;
    }), edges.end());
    std::unordered_set<uint64_t> connected;
    for (const auto& edge : edges) {
        connected.insert(edge.state.first_id);
        connected.insert(edge.state.second_id);
    }
    for (auto it = network.nodes.begin(); it != network.nodes.end();) {
        if (live.count(it->first) == 0 || connected.count(it->first) == 0) {
            it = network.nodes.erase(it);
        } else {
            ++it;
        }
    }
}

void retireNonDem(FluidParticles& p, const APICSolverParams& params) {
    if (p.dem_bonds.nodes.empty()) {
        return;
    }
    std::vector<uint64_t> live;
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (i >= p.particle_id.size() || i >= p.constitutive_model.size() ||
            (i < p.flags.size() && (p.flags[i] & kParticleFlagFrozen) != 0) ||
            (i < p.mass_fraction.size() && p.mass_fraction[i] <= 0.0f)) {
            continue;
        }
        const auto model = static_cast<MatterConstitutiveModel>(p.constitutive_model[i]);
        const uint32_t tag = i < p.substance_tag.size() ? p.substance_tag[i] : 0;
        if (model == MatterConstitutiveModel::Granular &&
            substanceTransportOwner(tag, model, params) == MatterTransportOwner::Grain) {
            live.push_back(p.particle_id[i]);
        }
    }
    retain(p.dem_bonds, live);
}

bool removeBody(Network& network, uint64_t body, std::string& error) {
    error.clear();
    std::vector<uint64_t> keep;
    bool found = false;
    for (const auto& [id, node] : network.nodes) {
        if (node.body == body) {
            found = true;
        } else {
            keep.push_back(id);
        }
    }
    if (!found || body == 0) {
        error = "DEM body: body not found";
        return false;
    }
    retain(network, keep);
    return true;
}

nlohmann::json encode(const Network& n) {
    auto nodes = nlohmann::json::array();
    auto edges = nlohmann::json::array();
    for (const auto& [id, node] : n.nodes) {
        nodes.push_back({id, node.body, node.orientation});
    }
    for (const auto& e : n.edges) {
        const auto& s = e.state;
        const auto& m = e.material;
        edges.push_back({s.first_id, s.second_id, s.rest_separation,
            s.first_rest_orientation, s.second_rest_orientation, s.rest_length_m,
            static_cast<uint32_t>(s.failure), s.fracture_dissipation_j,
            m.young_pa, m.poisson, m.radius_m, m.tensile_strength_pa,
            m.shear_strength_pa, e.elastic_energy_j});
    }
    return {{"version", 1}, {"nodes", nodes}, {"edges", edges},
        {"next_body", n.next_body}, {"retired_energy_j", n.retired_energy_j}};
}

bool validate(const Network& n, std::string& error) {
    error.clear();
    if (n.nodes.size() > maximum_edges || n.edges.size() > maximum_edges ||
        n.next_body == 0 || !std::isfinite(n.retired_energy_j) || n.retired_energy_j < 0.0) {
        error = "DEM graph: invalid capacity, allocator or energy";
        return false;
    }
    for (const auto& [id, node] : n.nodes) {
        double norm = 0.0;
        for (double v : node.orientation) {
            norm += v * v;
        }
        if (id == 0 || node.body == 0 || node.body >= n.next_body ||
            !std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-4) {
            error = "DEM graph: invalid node";
            return false;
        }
    }
    std::set<std::pair<uint64_t, uint64_t>> pairs;
    std::map<uint64_t, std::size_t> degrees;
    for (const auto& edge : n.edges) {
        const auto& s = edge.state;
        const auto a = n.nodes.find(s.first_id);
        const auto b = n.nodes.find(s.second_id);
        if (a == n.nodes.end() || b == n.nodes.end() ||
            a->second.body != b->second.body ||
            !pairs.emplace(std::min(s.first_id, s.second_id),
                           std::max(s.first_id, s.second_id)).second ||
            ++degrees[s.first_id] > maximum_degree || ++degrees[s.second_id] > maximum_degree ||
            !validEdge(edge, error)) {
            if (error.empty()) {
                error = "DEM graph: invalid topology/material";
            }
            return false;
        }
    }
    for (const auto& [id, node] : n.nodes) {
        if (degrees[id] == 0) {
            error = "DEM graph: node has no bond history";
            return false;
        }
    }
    return true;
}

bool decode(const nlohmann::json& j, Network& n, std::string& error) {
    error.clear();
    try {
        Network next;
        if (j.at("version") != 1 || !j.at("nodes").is_array() ||
            !j.at("edges").is_array() || j.at("nodes").size() > maximum_edges ||
            j.at("edges").size() > maximum_edges) {
            throw std::runtime_error("unsupported version or graph capacity");
        }
        next.next_body = unsignedValue(j.at("next_body"));
        next.retired_energy_j = numberValue(j.at("retired_energy_j"));
        if (next.next_body == 0 || !std::isfinite(next.retired_energy_j) ||
            next.retired_energy_j < 0.0) {
            throw std::runtime_error("invalid allocator/energy");
        }
        for (const auto& row : j.at("nodes")) {
            if (!row.is_array() || row.size() != 3) {
                throw std::runtime_error("invalid node shape");
            }
            const auto id = unsignedValue(row.at(0));
            Node node{unsignedValue(row.at(1)), arrayValue<4>(row.at(2))};
            double norm = 0.0;
            for (double v : node.orientation) {
                norm += v * v;
            }
            if (id == 0 || node.body == 0 || node.body >= next.next_body ||
                !std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-4 ||
                !next.nodes.emplace(id, node).second) {
                throw std::runtime_error("invalid or duplicate node");
            }
        }
        std::set<std::pair<uint64_t, uint64_t>> pairs;
        std::map<uint64_t, std::size_t> degrees;
        for (const auto& row : j.at("edges")) {
            if (!row.is_array() || row.size() != 14 || unsignedValue(row.at(6)) > 2) {
                throw std::runtime_error("invalid edge shape/failure state");
            }
            Edge e;
            auto& s = e.state;
            auto& m = e.material;
            s.first_id = unsignedValue(row.at(0));
            s.second_id = unsignedValue(row.at(1));
            s.rest_separation = arrayValue<3>(row.at(2));
            s.first_rest_orientation = arrayValue<4>(row.at(3));
            s.second_rest_orientation = arrayValue<4>(row.at(4));
            s.rest_length_m = numberValue(row.at(5));
            s.failure = static_cast<Failure>(unsignedValue(row.at(6)));
            s.fracture_dissipation_j = numberValue(row.at(7));
            m = {numberValue(row.at(8)), numberValue(row.at(9)), numberValue(row.at(10)),
                 numberValue(row.at(11)), numberValue(row.at(12))};
            e.elastic_energy_j = numberValue(row.at(13));
            const auto a = next.nodes.find(s.first_id);
            const auto b = next.nodes.find(s.second_id);
            if (a == next.nodes.end() || b == next.nodes.end() ||
                a->second.body != b->second.body ||
                !pairs.emplace(std::min(s.first_id, s.second_id),
                               std::max(s.first_id, s.second_id)).second ||
                ++degrees[s.first_id] > maximum_degree ||
                ++degrees[s.second_id] > maximum_degree || !validEdge(e, error)) {
                throw std::runtime_error("invalid bond topology/material");
            }
            next.edges.push_back(e);
        }
        if (!validate(next, error)) {
            return false;
        }
        n = std::move(next);
        return true;
    } catch (const std::exception& e) {
        error = std::string("DEM bond snapshot: ") + e.what();
        return false;
    }
}

nlohmann::json summary(const Network& n) {
    std::set<uint64_t> bodies;
    std::size_t broken = 0;
    double elastic = 0.0;
    double dissipated = n.retired_energy_j;
    for (const auto& [id, node] : n.nodes) {
        bodies.insert(node.body);
    }
    for (const auto& e : n.edges) {
        broken += e.state.failure != Failure::None;
        elastic += e.elastic_energy_j;
        dissipated += e.state.fracture_dissipation_j;
    }
    return {{"bodies", bodies.size()}, {"body_ids", std::vector<uint64_t>(bodies.begin(), bodies.end())},
        {"particles", n.nodes.size()},
        {"bonds", n.edges.size()}, {"broken", broken}, {"elastic_energy_j", elastic},
        {"dissipated_energy_j", dissipated}};
}
} // namespace RayTrophiSim::Fluid::DemBond
