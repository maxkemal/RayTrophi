#include "RayFusion/ProbeField.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace RayFusion {
namespace {
std::atomic<uint64_t> nextFieldId{1};
bool validGrid(const Grid& grid, std::string& error) {
    uint64_t total = 1;
    if (!std::isfinite(grid.spacing) || grid.spacing <= 0 ||
        grid.targetUpdates == 0 || grid.targetUpdates > 1024) {
        error = "spacing must be finite and positive; targetUpdates must be 1..1024";
        return false;
    }
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (grid.counts[axis] == 0 || grid.counts[axis] > 128) {
            error = "each probe grid dimension must be 1..128";
            return false;
        }
        total *= grid.counts[axis];
        const auto lo = grid.minimum[axis];
        const double maxWorldCoordinate = (std::max)(std::abs(static_cast<double>(lo)),
            std::abs(static_cast<double>(lo) + grid.counts[axis])) * grid.spacing;
        if (lo < -1000000000LL || lo > 1000000000LL ||
            !std::isfinite(maxWorldCoordinate) ||
            maxWorldCoordinate > (std::numeric_limits<float>::max)() / 4.0) {
            error = "probe grid world coordinates exceed the supported range";
            return false;
        }
    }
    if (total > 32768) {
        error = "probe grid exceeds the 32768 probe memory bound";
        return false;
    }
    return true;
}
bool validRevision(const Revision& revision, std::string& error) {
    if (!revision.deviceEpoch || !revision.sceneEpoch) {
        error = "nonzero device and scene epochs are required";
        return false;
    }
    return true;
}
bool validPacket(const ProbePacket& packet) {
    for (const auto& texel : packet) {
        for (float value : texel.irradiance)
            if (!std::isfinite(value) || value < 0) return false;
        for (float value : texel.distance)
            if (!std::isfinite(value) || value < 0) return false;
        const double mean = texel.distance[0];
        const double mean2 = mean * mean;
        const double second = texel.distance[1];
        if (second + 1e-5 * (std::max)(mean2, 1.0) < mean2) return false;
    }
    return true;
}
} // namespace

bool Revision::operator==(const Revision& rhs) const {
    return deviceEpoch == rhs.deviceEpoch && sceneEpoch == rhs.sceneEpoch &&
        geometry == rhs.geometry && lighting == rhs.lighting;
}

ProbeField::ProbeField() : m_fieldId(nextFieldId.fetch_add(1, std::memory_order_relaxed)) {}

Budget budgetForQuality(const std::string& quality) {
    if (quality == "performance") return {32, 8, 256};
    if (quality == "quality") return {128, 32, 4096};
    if (quality == "full") return {256, 32, 8192};
    return {64, 16, 1024}; // auto and balanced, an initial bound, not a timing claim
}

bool ProbeField::configure(const Grid& grid, Revision revision, std::string& error) {
    error.clear();
    if (!validGrid(grid, error) || !validRevision(revision, error)) return false;
    // Allocate before mutating so allocation failure leaves the old field intact.
    std::vector<Slot> slots(std::size_t(grid.counts[0]) * grid.counts[1] * grid.counts[2]);
    m_grid = grid;
    m_revision = revision;
    m_slots.swap(slots);
    ++m_generation;
    m_cursor = 0;
    m_accepted = m_rejected = 0;
    assignCells(true);
    return true;
}

uint32_t ProbeField::slotFor(const Cell& cell) const {
    uint32_t coords[3];
    for (unsigned a = 0; a < 3; ++a) {
        const int64_t n = m_grid.counts[a];
        coords[a] = static_cast<uint32_t>((cell[a] % n + n) % n);
    }
    return (coords[2] * m_grid.counts[1] + coords[1]) * m_grid.counts[0] + coords[0];
}

bool ProbeField::contains(const Cell& cell) const {
    if (m_slots.empty()) return false;
    for (unsigned a = 0; a < 3; ++a)
        if (cell[a] < m_grid.minimum[a] ||
            cell[a] >= m_grid.minimum[a] + m_grid.counts[a]) return false;
    return true;
}

void ProbeField::assignCells(bool clearAll) {
    for (uint32_t z = 0; z < m_grid.counts[2]; ++z)
        for (uint32_t y = 0; y < m_grid.counts[1]; ++y)
            for (uint32_t x = 0; x < m_grid.counts[0]; ++x) {
                const Cell cell{m_grid.minimum[0] + x, m_grid.minimum[1] + y,
                                m_grid.minimum[2] + z};
                auto& slot = m_slots[slotFor(cell)];
                if (clearAll || slot.cell != cell) {
                    slot = {};
                    slot.cell = cell;
                }
            }
}

bool ProbeField::scroll(const Cell& minimum, std::string& error) {
    error.clear();
    if (m_slots.empty()) { error = "probe field is not configured"; return false; }
    Grid next = m_grid;
    next.minimum = minimum;
    if (!validGrid(next, error)) return false;
    m_grid = next;
    // Tickets for retained world cells remain valid. Recycled slots reject old
    // tickets by cell + serial, including an out-and-back camera movement.
    assignCells(false);
    return true;
}

bool ProbeField::invalidate(Revision revision, std::string& error) {
    error.clear();
    if (m_slots.empty()) { error = "probe field is not configured"; return false; }
    if (!validRevision(revision, error)) return false;
    if (revision == m_revision) return true;
    m_revision = revision;
    ++m_generation;
    m_cursor = 0;
    assignCells(true);
    return true;
}

std::vector<Ticket> ProbeField::schedule(const Budget& budget) {
    std::vector<Ticket> batch;
    if (m_slots.empty() || !budget.raysPerProbe) return batch;
    const auto count = static_cast<uint32_t>(m_slots.size());
    const uint32_t limit = (std::min)({budget.maxProbes, budget.maxRays / budget.raysPerProbe, count});
    batch.reserve(limit);
    for (uint32_t scanned = 0; scanned < count && batch.size() < limit; ++scanned) {
        const uint32_t index = m_cursor;
        m_cursor = (m_cursor + 1) % count;
        auto& slot = m_slots[index];
        if (slot.inFlight || slot.updates >= m_grid.targetUpdates) continue;
        slot.serial = ++m_serial;
        slot.inFlight = true;
        batch.push_back({m_fieldId, m_generation, slot.serial, index, slot.cell, m_revision});
    }
    return batch;
}

bool ProbeField::matches(const Ticket& ticket) const {
    if (ticket.field != m_fieldId || ticket.generation != m_generation || !(ticket.revision == m_revision) ||
        ticket.slot >= m_slots.size()) return false;
    const auto& slot = m_slots[ticket.slot];
    return slot.inFlight && slot.cell == ticket.cell && slot.serial == ticket.serial;
}

bool ProbeField::publish(const Ticket& ticket, const ProbePacket& packet,
                         float historyWeight, std::string& error) {
    error.clear();
    if (!matches(ticket)) {
        ++m_rejected;
        error = "stale, cancelled or already completed probe ticket";
        return false;
    }
    if (!std::isfinite(historyWeight) || historyWeight < 0 || historyWeight > 0.98f ||
        !validPacket(packet)) {
        ++m_rejected;
        error = "invalid radiance, distance moments or history weight; packet was not published";
        return false;
    }
    auto& slot = m_slots[ticket.slot];
    const float history = slot.valid ? historyWeight : 0.0f;
    for (uint32_t t = 0; t < kProbeTexels; ++t)
        for (unsigned c = 0; c < 4; ++c) {
            slot.packet[t].irradiance[c] = history * slot.packet[t].irradiance[c] +
                (1.0f - history) * packet[t].irradiance[c];
            slot.packet[t].distance[c] = history * slot.packet[t].distance[c] +
                (1.0f - history) * packet[t].distance[c];
        }
    ++slot.updates;
    slot.valid = true;
    slot.inFlight = false;
    ++m_accepted;
    return true;
}

bool ProbeField::cancel(const Ticket& ticket) {
    if (!matches(ticket)) return false;
    m_slots[ticket.slot].inFlight = false;
    return true;
}

const ProbePacket* ProbeField::lookup(const Cell& cell) const {
    if (!contains(cell)) return nullptr;
    const auto& slot = m_slots[slotFor(cell)];
    return slot.valid && slot.cell == cell ? &slot.packet : nullptr;
}

Stats ProbeField::stats() const {
    Stats out;
    out.total = static_cast<uint32_t>(m_slots.size());
    out.accepted = m_accepted;
    out.rejected = m_rejected;
    for (const auto& slot : m_slots) {
        out.valid += slot.valid ? 1u : 0u;
        out.inFlight += slot.inFlight ? 1u : 0u;
        out.pending += !slot.inFlight && slot.updates < m_grid.targetUpdates ? 1u : 0u;
    }
    return out;
}

float momentVisibility(float mean, float meanSquare, float distance) {
    if (!std::isfinite(mean) || !std::isfinite(meanSquare) || !std::isfinite(distance) ||
        mean < 0 || meanSquare < 0 || distance < 0) return 0;
    if (double(meanSquare) + 1e-5 * (std::max)(double(mean) * mean, 1.0) <
        double(mean) * mean) return 0;
    if (distance <= mean) return 1;
    const double variance = (std::max)(double(meanSquare) - double(mean) * mean, 0.0);
    const double delta = double(distance) - mean;
    const double weight = variance / (variance + delta * delta);
    return static_cast<float>(weight * weight * weight);
}
} // namespace RayFusion
