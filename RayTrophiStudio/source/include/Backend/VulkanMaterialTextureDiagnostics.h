#pragma once

#include "VulkanMaterialTextureDescriptors.h"
#include "VulkanMaterialTextureUploadQueue.h"
#include "../json.hpp"

namespace Backend::MaterialTexturePages {

inline nlohmann::json uploadQueueDiagnostics(const UploadQueueStatus& status) {
    return {
        {"pending_uploads", status.pendingUploads},
        {"completed_upload_serial", status.completedUploadSerial},
        {"upload_not_ready_rejections", status.notReadyRejections}
    };
}

inline nlohmann::json descriptorRejectionDiagnostics(const DescriptorPreparation& attempt) {
    using nlohmann::json;
    const char* reason = "unknown";
    switch (attempt.reason) {
    case DescriptorRejection::None: return nullptr;
    case DescriptorRejection::InvalidDummy: reason = "invalid_dummy"; break;
    case DescriptorRejection::InvalidPage: reason = "invalid_page"; break;
    case DescriptorRejection::LayerStateOverCapacity: reason = "layer_state_over_capacity"; break;
    case DescriptorRejection::PageOutOfRange: reason = "page_out_of_range"; break;
    case DescriptorRejection::MissingPage: reason = "missing_page"; break;
    case DescriptorRejection::LayerOutOfRange: reason = "layer_out_of_range"; break;
    case DescriptorRejection::LayerUnloaded: reason = "layer_unloaded"; break;
    case DescriptorRejection::LayerRetired: reason = "layer_retired"; break;
    }
    return {
        {"reason", reason},
        {"page", attempt.page ? json(*attempt.page) : json(nullptr)},
        {"layer", attempt.layer ? json(*attempt.layer) : json(nullptr)},
        {"reference_index", attempt.referenceIndex ? json(*attempt.referenceIndex) : json(nullptr)},
        {"reference", attempt.reference ? json(*attempt.reference) : json(nullptr)}
    };
}

struct DescriptorPublicationDiagnostics {
    uint64_t rejectedPublications = 0;
    DescriptorPreparation lastRejection;

    // Call once per rejected frame publication, on the residency owner thread.
    template <typename Log>
    void recordRejection(const DescriptorPreparation& attempt, Log log) {
        if (attempt) {
            return;
        }
        ++rejectedPublications;
        lastRejection = attempt;
        log(descriptorRejectionDiagnostics(attempt).dump());
    }

    nlohmann::json snapshot() const {
        return {
            {"descriptor_publications_rejected", rejectedPublications},
            {"last_descriptor_rejection", descriptorRejectionDiagnostics(lastRejection)}
        };
    }
};

inline nlohmann::json layerDiagnostics(const PublishedPages& pages,
                                       const std::vector<uint32_t>& references) {
    uint64_t used = 0;
    uint64_t capacity = 0;
    bool valid = true;
    for (const PublishedPage& page : pages) {
        used += std::count(page.layers.begin(), page.layers.end(), LayerState::Resident);
        if (page.image) {
            capacity += page.image->layers;
        }
        valid = checkPublishedPage(page) == DescriptorRejection::None && valid;
    }
    const bool referencesValid = std::all_of(references.begin(), references.end(),
                                            [&](uint32_t ref) {
        return checkPublishedTexture(pages, ref) == DescriptorRejection::None;
    });
    return {
        {"layers_used", used},
        {"layers_capacity", capacity},
        {"layers_valid", valid && used <= capacity && referencesValid}
    };
}

// Common status fragment for the Vulkan texture residency snapshot.
inline nlohmann::json imageDiagnostics(const ImageLimits& limits, const Plan& plan) {
    using nlohmann::json;
    json queries = json::array();
    for (std::size_t i = 0; i < limits.profiles.size(); ++i) {
        const auto& entry = limits.profiles[i];
        json properties = nullptr;
        if (entry.result == VK_SUCCESS) {
            const auto& image = entry.properties;
            properties = {
                {"max_width", image.maxExtent.width},
                {"max_height", image.maxExtent.height},
                {"max_mip_levels", image.maxMipLevels},
                {"max_array_layers", image.maxArrayLayers},
                {"max_resource_bytes", image.maxResourceSize},
                {"sample_counts", image.sampleCounts}
            };
        }
        queries.push_back({
            {"profile", i},
            {"format", static_cast<int32_t>(entry.profile.format)},
            {"usage_flags", entry.profile.usage},
            {"create_flags", entry.profile.flags},
            {"vk_result", static_cast<int32_t>(entry.result)},
            {"properties", std::move(properties)}
        });
    }
    json warnings = json::array();
    json rejected = json::array();
    for (const RejectedTexture& texture : plan.rejectedTextures) {
        json rejection = {
            {"code", "unsupported_image"},
            {"texture_index", texture.texture},
            {"profile", texture.key.compatibility},
            {"width", texture.key.width},
            {"height", texture.key.height},
            {"mip_levels", texture.key.mipLevels}
        };
        if (texture.key.compatibility < limits.profiles.size()) {
            const auto& entry = limits.profiles[texture.key.compatibility];
            rejection["format"] = static_cast<int32_t>(entry.profile.format);
            rejection["usage_flags"] = entry.profile.usage;
            rejection["vk_result"] = static_cast<int32_t>(entry.result);
        }
        warnings.push_back(rejection);
        rejected.push_back(std::move(rejection));
    }
    return {
        {"image_format_queries", std::move(queries)},
        {"rejected_textures", std::move(rejected)},
        {"warnings", std::move(warnings)}
    };
}

} // namespace Backend::MaterialTexturePages
