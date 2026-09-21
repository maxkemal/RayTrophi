#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace RayFusion {

// CPU control-plane foundation. Only a completed, matching producer ticket
// may publish lighting. No renderer/device ownership is inferred from names.
struct Revision {
    uint64_t deviceEpoch = 0;
    uint64_t sceneEpoch = 0;
    uint64_t geometry = 0;
    uint64_t lighting = 0; // includes material, texture, sky and emissive changes
    bool operator==(const Revision& rhs) const;
};
using Cell = std::array<int64_t, 3>;

struct Grid {
    std::array<uint32_t, 3> counts{8, 4, 8};
    Cell minimum{0, 0, 0};
    float spacing = 2.0f;
    uint32_t targetUpdates = 8;
};

// GPU bound, not a design opinion: the texel buffer is allocated ONCE for this
// many slots so changing the grid never reallocates a buffer an in-flight frame
// is still reading. 1024 slots * 64 texels * 32 B = 2 MiB.
// ★★★★ 2026-09-15'te 1024'ten 4096'ya cikarildi, ve gerekce bir OLCUM:
//   elle oturtulmus bir oda izgarasi (spacing 2,0 / 10x6x14) tavanin 840'ini
//   yiyordu -- yani "daha fazla prob" istegi bir tercih degil, ARTIK TAVANA
//   DAYANMISTI. 4096 * 64 texel * 32 B = 8 MiB; alan bir kez ayrilir ve
//   izgara degisimi bufferi yeniden tahsis etmez, bu tavanin butun amaci bu.
// ★★★ Tavani yukseltmek izgarayi BUYUTMEZ, yalnizca izin verir. Gercek
//   yogunlugu `auto_fit` secer ve `rayfusion.probe_field`de `total` olarak
//   okunur -- tavan ile kullanilan ayri sayilardir ve oyle kalmalidir.
constexpr uint32_t kMaxProbeSlots = 4096;

// A PARTIAL grid edit. Every field is optional: a caller that only wants to move
// the window must not have to restate its shape, and restating it is how a
// caller silently resets a count it never meant to touch. What is not sent keeps
// what the field already has.
//
// `center` is world units and is resolved against the FINAL spacing and counts,
// so it is resolved by the field's owner, never by the caller -- a caller that
// converted metres to cells itself would be converting with the OLD spacing in
// the same call that changes it.
struct GridRequest {
    bool hasCounts = false;  std::array<uint32_t, 3> counts{};
    bool hasSpacing = false; float spacing = 0.0f;
    bool hasMinimum = false; std::array<int32_t, 3> minimum{};
    bool hasCenter = false;  std::array<float, 3> center{};
    // Otomatik yerlesimi acar/kapatir. Ayri bir alan, cunku "gonderilmedi" ile
    // "false gonderildi" ayni sey degil: her kismi izgara duzenlemesi otomatigi
    // sessizce kapatsaydi, yalnizca overlay acmak bile yerlesimi dondururdu.
    bool hasAutoFit = false; bool autoFit = false;
};

struct Budget {
    uint32_t raysPerProbe = 64;
    uint32_t maxProbes = 16;
    uint32_t maxRays = 1024;
};
Budget budgetForQuality(const std::string& quality);

constexpr uint32_t kProbeSide = 8;
constexpr uint32_t kProbeTexels = kProbeSide * kProbeSide;
constexpr uint32_t kProbeAbiVersion = 1;
// Shared tracing horizon, also published to the specular visibility consumer.
constexpr float kProbeTraceDistance = 200.0f;
// Scene-linear incoming diffuse radiance (irradiance / PI), not albedo or
// display colour. Distance moments use world units and world units squared.
// Octahedral texel ordering: y * kProbeSide + x.
struct alignas(16) ProbeTexel {
    std::array<float, 4> irradiance{};
    std::array<float, 4> distance{}; // mean, mean-square; remaining lanes reserved
};
static_assert(sizeof(ProbeTexel) == 32, "RayFusion probe GLSL ABI");
static_assert(offsetof(ProbeTexel, distance) == 16, "RayFusion distance lane ABI");
using ProbePacket = std::array<ProbeTexel, kProbeTexels>;

struct Ticket {
    uint64_t field = 0;
    uint64_t generation = 0;
    uint64_t serial = 0;
    uint32_t slot = 0;
    Cell cell{};
    Revision revision;
};
struct Stats {
    uint32_t total = 0;
    uint32_t valid = 0;
    uint32_t inFlight = 0;
    uint32_t pending = 0;
    uint64_t accepted = 0;
    uint64_t rejected = 0;
};

// Single owner thread (future RayFusion scene service). GPU completion records
// must be marshalled back to that owner; this class is deliberately not a lock
// around live scene geometry. Scrolling scans this bounded grid, never triangles.
class ProbeField {
public:
    ProbeField();
    ProbeField(const ProbeField&) = delete;
    ProbeField& operator=(const ProbeField&) = delete;
    bool configure(const Grid& grid, Revision revision, std::string& error);
    bool scroll(const Cell& minimum, std::string& error);
    bool invalidate(Revision revision, std::string& error);
    std::vector<Ticket> schedule(const Budget& budget);
    bool publish(const Ticket& ticket, const ProbePacket& packet,
                 float historyWeight, std::string& error);
    bool cancel(const Ticket& ticket);
    const ProbePacket* lookup(const Cell& cell) const;
    // The GPU consumer needs the same cell -> slot mapping to index its buffer.
    // Exposed rather than reimplemented: a second copy of the toroidal hash
    // would read a different slot than the one that was published, and the
    // symptom would be a wrong ambient value, not an error.
    uint32_t slotIndex(const Cell& cell) const { return slotFor(cell); }
    Stats stats() const;
    const Grid& grid() const { return m_grid; }
private:
    struct Slot {
        Cell cell{};
        uint64_t serial = 0;
        uint32_t updates = 0;
        bool valid = false;
        bool inFlight = false;
        ProbePacket packet{};
    };
    uint32_t slotFor(const Cell& cell) const;
    bool contains(const Cell& cell) const;
    bool matches(const Ticket& ticket) const;
    void assignCells(bool clearAll);
    Grid m_grid;
    const uint64_t m_fieldId;
    Revision m_revision;
    std::vector<Slot> m_slots;
    uint64_t m_generation = 0;
    uint64_t m_serial = 0;
    uint64_t m_accepted = 0;
    uint64_t m_rejected = 0;
    uint32_t m_cursor = 0;
};

// Bounded moment-based visibility estimate; not proof of exact occlusion.
float momentVisibility(float mean, float meanSquare, float distance);

struct Check { std::string name; bool passed; std::string detail; };
// Runs isolated fixtures against the real control-plane implementation. Does
// not touch the scene, GPU, viewport mode, settings or production probe cache.
std::vector<Check> validateProbeCore();

} // namespace RayFusion
