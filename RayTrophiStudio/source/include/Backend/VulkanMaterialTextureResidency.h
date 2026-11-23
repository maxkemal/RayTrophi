#pragma once

#include "VulkanMaterialTextureDescriptors.h"

#include <limits>

namespace Backend::MaterialTexturePages {

struct LayerReservation {
    uint32_t ref = kNoTexture;
    uint64_t version = 0;
};

template <typename Snapshot>
class FramePublication;

// Render-thread state for one array page. Upload and frame-reader serials are
// separate monotonic completion domains, even when they share a Vulkan queue.
class PageResidency {
public:
    PageResidency(uint32_t page, uint32_t capacity)
        : m_page(page), m_layers(page < kPageBudget && capacity <= 65535u ? capacity : 0) {}

    LayerReservation reserve() {
        if (m_version == (std::numeric_limits<uint64_t>::max)()) {
            return {};
        }
        for (uint32_t i = 0; i < m_layers.size(); ++i) {
            Layer& layer = m_layers[i];
            if (layer.phase == Phase::Free) {
                layer = {Phase::Reserved, ++m_version, 0};
                return {packRef(m_page, i), layer.version};
            }
        }
        return {};
    }

    bool markUploadSubmitted(LayerReservation reservation, uint64_t serial) {
        Layer* layer = find(reservation);
        if (!layer || layer->phase != Phase::Reserved || !serial) {
            return false;
        }
        layer->phase = Phase::Uploading;
        layer->completion = serial;
        return true;
    }

    // On rejection use discard(), keeping the existing material/descriptor pair.
    bool canPublish(LayerReservation reservation, uint64_t completedSerial) const {
        const Layer* layer = find(reservation);
        return layer && layer->phase == Phase::Uploading && layer->completion <= completedSerial;
    }

    // Handles a failed submission or stale revision. A submitted upload still
    // owns its destination until the upload fence signals.
    bool discard(LayerReservation reservation) {
        Layer* layer = find(reservation);
        if (!layer) {
            return false;
        }
        if (layer->phase == Phase::Reserved) {
            *layer = {};
            return true;
        }
        if (layer->phase == Phase::Uploading) {
            layer->phase = Phase::DiscardedUpload;
            return true;
        }
        return false;
    }

    void reclaim(uint64_t completedUploadSerial, uint64_t completedReaderSerial) {
        for (Layer& layer : m_layers) {
            const bool readerDone =
                layer.phase == Phase::Retired && layer.completion <= completedReaderSerial;
            const bool uploadDone =
                layer.phase == Phase::DiscardedUpload && layer.completion <= completedUploadSerial;
            if (readerDone || uploadDone) {
                layer = {};
            }
        }
    }

    // Include a ready upload when preparing the candidate descriptor snapshot.
    // This projection leaves the currently published residency unchanged.
    std::vector<LayerState> describe(LayerReservation ready = {},
                                     uint64_t completedSerial = 0,
                                     LayerReservation retiring = {}) const {
        const bool includeReady = canPublish(ready, completedSerial);
        const Layer* retiringLayer = find(retiring);
        const uint32_t readyIndex = ready.ref & 0xffffu;
        std::vector<LayerState> states;
        states.reserve(m_layers.size());
        for (uint32_t i = 0; i < m_layers.size(); ++i) {
            const Phase phase = m_layers[i].phase;
            if (phase == Phase::Retired ||
                (retiringLayer == &m_layers[i] && phase == Phase::Resident)) {
                states.push_back(LayerState::Retired);
            } else if (phase == Phase::Resident || (includeReady && i == readyIndex)) {
                states.push_back(LayerState::Resident);
            } else {
                states.push_back(LayerState::Unloaded);
            }
        }
        return states;
    }

private:
    template <typename Snapshot>
    friend class FramePublication;

    enum class Phase : uint8_t {
        Free,
        Reserved,
        Uploading,
        Resident,
        Retired,
        DiscardedUpload
    };

    struct Layer {
        Phase phase = Phase::Free;
        uint64_t version = 0;
        uint64_t completion = 0;
    };

    const Layer* find(LayerReservation reservation) const {
        const uint32_t layer = reservation.ref & 0xffffu;
        if ((reservation.ref >> 16) != m_page || layer >= m_layers.size() ||
            !reservation.version || m_layers[layer].version != reservation.version) {
            return nullptr;
        }
        return &m_layers[layer];
    }

    Layer* find(LayerReservation reservation) {
        return const_cast<Layer*>(static_cast<const PageResidency&>(*this).find(reservation));
    }

    bool canRetire(LayerReservation previous) const {
        if (previous.ref == kNoTexture) {
            return true;
        }
        const Layer* layer = find(previous);
        return layer && layer->phase == Phase::Resident;
    }

    void commitRetirement(LayerReservation previous, uint64_t previousLastReader) {
        if (Layer* old = find(previous)) {
            old->phase = Phase::Retired;
            old->completion = previousLastReader;
        }
    }

    void commitPublication(LayerReservation replacement) {
        find(replacement)->phase = Phase::Resident;
    }

    uint32_t m_page;
    uint64_t m_version = 0;
    std::vector<Layer> m_layers;
};

} // namespace Backend::MaterialTexturePages
