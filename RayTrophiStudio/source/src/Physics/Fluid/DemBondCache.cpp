#include "Fluid/DemBondCache.h"
#include <istream>
#include <ostream>
#include <type_traits>
#include <unordered_set>

namespace RayTrophiSim::Fluid::DemBond {
namespace {
constexpr uint64_t max_bytes = 256u * 1024u * 1024u;
struct Header {
    uint64_t nodes = 0;
    uint64_t edges = 0;
    uint64_t next_body = 1;
    double retired_energy = 0.0;
};
struct DiskNode {
    uint64_t id = 0;
    uint64_t body = 0;
    Quaternion orientation{};
};
struct DiskEdge {
    uint64_t first = 0;
    uint64_t second = 0;
    Vector separation{};
    Quaternion qa{};
    Quaternion qb{};
    double length = 0.0;
    uint64_t failure = 0;
    double dissipated = 0.0;
    Material material;
    double elastic = 0.0;
};
static_assert(sizeof(Header) == 32 && sizeof(DiskNode) == 48 && sizeof(DiskEdge) == 176);
static_assert(std::is_trivially_copyable_v<DiskEdge>);

bool allowed(const Header& h) {
    return h.nodes <= 4000000u && h.edges <= 4000000u &&
        sizeof(Header) + h.nodes * sizeof(DiskNode) + h.edges * sizeof(DiskEdge) <= max_bytes;
}

template <class T>
void write(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <class T>
bool read(std::istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(stream);
}
}

bool writeCache(std::ostream& stream, const Network& network) {
    try {
        const Header header{network.nodes.size(), network.edges.size(), network.next_body,
                            network.retired_energy_j};
        if (!allowed(header)) {
            return false;
        }
        write(stream, header);
        for (const auto& [id, node] : network.nodes) {
            write(stream, DiskNode{id, node.body, node.orientation});
        }
        for (const auto& edge : network.edges) {
            const auto& s = edge.state;
            const DiskEdge row{s.first_id, s.second_id, s.rest_separation,
                s.first_rest_orientation, s.second_rest_orientation, s.rest_length_m,
                static_cast<uint64_t>(s.failure), s.fracture_dissipation_j,
                edge.material, edge.elastic_energy_j};
            write(stream, row);
        }
        return static_cast<bool>(stream);
    } catch (...) {
        return false;
    }
}

bool readCache(std::istream& stream, const std::vector<uint64_t>& particle_ids,
               Network& network) {
    try {
        Header header;
        if (!read(stream, header) || !allowed(header)) {
            return false;
        }
        Network next;
        next.next_body = header.next_body;
        next.retired_energy_j = header.retired_energy;
        const std::unordered_set<uint64_t> ids(particle_ids.begin(), particle_ids.end());
        for (uint64_t i = 0; i < header.nodes; ++i) {
            DiskNode row;
            if (!read(stream, row) || ids.count(row.id) == 0 ||
                !next.nodes.emplace(row.id, Node{row.body, row.orientation}).second) {
                return false;
            }
        }
        next.edges.reserve(static_cast<std::size_t>(header.edges));
        for (uint64_t i = 0; i < header.edges; ++i) {
            DiskEdge row;
            if (!read(stream, row) || row.failure > 2u) {
                return false;
            }
            Edge edge;
            edge.state.first_id = row.first;
            edge.state.second_id = row.second;
            edge.state.rest_separation = row.separation;
            edge.state.first_rest_orientation = row.qa;
            edge.state.second_rest_orientation = row.qb;
            edge.state.rest_length_m = row.length;
            edge.state.failure = static_cast<Failure>(row.failure);
            edge.state.fracture_dissipation_j = row.dissipated;
            edge.material = row.material;
            edge.elastic_energy_j = row.elastic;
            next.edges.push_back(edge);
        }
        std::string error;
        if (!validate(next, error)) {
            return false;
        }
        network = std::move(next);
        return true;
    } catch (...) {
        return false;
    }
}
} // namespace RayTrophiSim::Fluid::DemBond
