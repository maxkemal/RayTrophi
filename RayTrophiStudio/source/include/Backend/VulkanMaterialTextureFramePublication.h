#pragma once

#include "VulkanMaterialTextureDiagnostics.h"
#include "VulkanMaterialTextureResidency.h"

#include <deque>

namespace Backend::MaterialTexturePages {

struct FrameReaderSerial {
    uint64_t value = 0;
};

enum class FramePublicationResult {
    Published,
    StaleRevision,
    UploadPending,
    InvalidResidency,
    InvalidDescriptors,
    ResourcePreparationFailed
};

// Render-thread state. Snapshot owns the material buffer, descriptor set,
// samplers and array images (including the dummy) used by a draw. Callbacks
// must not reenter publication, change residency or submit readers.
template <typename Snapshot>
class FramePublication {
public:
    template <typename Allocate, typename Prepare, typename Log>
    FramePublicationResult publish(uint64_t revision, uint64_t wantedRevision,
                                   PageResidency& residency, LayerReservation replacement,
                                   LayerReservation previous, const UploadQueueStatus& uploads,
                                   PublishedPages pages,
                                   const std::vector<uint32_t>& references,
                                   const ArrayImage& dummy, VkSampler dummySampler,
                                   Allocate allocateFreshSnapshot, Prepare prepare, Log log,
                                   PageResidency* previousResidency = nullptr) {
        if (!revision || revision != wantedRevision || revision <= m_revision) {
            return FramePublicationResult::StaleRevision;
        }
        PageResidency& retiring = previousResidency ? *previousResidency : residency;
        if ((&retiring != &residency &&
             (retiring.m_page >= pages.size() || retiring.m_page == residency.m_page)) ||
            !retiring.canRetire(previous)) {
            return FramePublicationResult::InvalidResidency;
        }
        if (!residency.canPublish(replacement, uploads.completedUploadSerial)) {
            return FramePublicationResult::UploadPending;
        }

        pages[residency.m_page].layers = residency.describe(
            replacement, uploads.completedUploadSerial,
            &retiring == &residency ? previous : LayerReservation{});
        if (&retiring != &residency) {
            pages[retiring.m_page].layers = retiring.describe({}, 0, previous);
        }
        DescriptorImages images;
        const auto attempt = prepareDescriptorImages(pages, references, dummy,
                                                     dummySampler, images);
        if (!attempt) {
            m_diagnostics.recordRejection(attempt, log);
            return FramePublicationResult::InvalidDescriptors;
        }

        // Allocate an owned descriptor set for every attempt that reaches
        // preparation. Check for aliases before prepare can write descriptors.
        std::shared_ptr<Snapshot> candidate = allocateFreshSnapshot();
        if (!candidate || !isFreshDescriptorSet(candidate->descriptorSet()) ||
            !prepare(*candidate, images)) {
            return FramePublicationResult::ResourcePreparationFailed;
        }

        // All fallible work is complete. Commit residency and the frame snapshot
        // together on the render thread before releasing the previous snapshot.
        std::shared_ptr<const Snapshot> next = std::move(candidate);
        retiring.commitRetirement(previous, m_lastSubmittedReader.value);
        residency.commitPublication(replacement);
        m_revision = revision;
        m_current.swap(next);
        return FramePublicationResult::Published;
    }

    std::shared_ptr<const Snapshot> current() const {
        return m_current;
    }

    // Associate the snapshot used by the command buffer with its reader fence.
    // Call in submission order; failed submissions need no lease.
    void retainReader(FrameReaderSerial serial, std::shared_ptr<const Snapshot> snapshot) {
        assert(serial.value && snapshot);
        assert(serial.value >= m_lastSubmittedReader.value);
        m_readers.push_back({serial, std::move(snapshot)});
        m_lastSubmittedReader = serial;
    }

    void reclaimReaders(FrameReaderSerial completed) {
        while (!m_readers.empty() && m_readers.front().serial.value <= completed.value) {
            m_readers.pop_front();
        }
    }

    const DescriptorPublicationDiagnostics& diagnostics() const {
        return m_diagnostics;
    }

private:
    struct Reader {
        FrameReaderSerial serial;
        std::shared_ptr<const Snapshot> snapshot;
    };

    bool isFreshDescriptorSet(VkDescriptorSet set) const {
        if (!set || (m_current && set == m_current->descriptorSet())) {
            return false;
        }
        return std::none_of(m_readers.begin(), m_readers.end(), [set](const Reader& reader) {
            return set == reader.snapshot->descriptorSet();
        });
    }

    uint64_t m_revision = 0;
    FrameReaderSerial m_lastSubmittedReader;
    std::shared_ptr<const Snapshot> m_current;
    std::deque<Reader> m_readers;
    DescriptorPublicationDiagnostics m_diagnostics;
};

} // namespace Backend::MaterialTexturePages
